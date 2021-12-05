import torch


from .assign_result import AssignResult
from .base_assigner import BaseAssigner
from mmdet.ops.point_justify import pointsJf

from ..iou_calculators import build_iou_calculator
from ..builder import BBOX_ASSIGNERS
# from ..geometry import rbox_overlaps
# from ..transforms_rbox import rbox2poly_torch, rbox2poly
from ..transforms_rotated import rotated_box_to_poly

@BBOX_ASSIGNERS.register_module
class SASAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): number of bbox selected in each level
    """

    def __init__(self,
                 topk,
                 iou_calculator=dict(type='BboxOverlaps2D')):
        self.topk = topk
        self.iou_calculator = build_iou_calculator(iou_calculator)

    # https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as postive
        6. limit the positive sample's center in gt


        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 4).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000
        bboxes = bboxes[:, :5]
        num_gt, num_bboxes = gt_bboxes.shape[0], bboxes.shape[0]
        # print('gt_bboxes', gt_bboxes.shape)
        # print('bboxes', bboxes.shape)

        if bboxes.shape[0] == 0 or gt_bboxes.shape[0] == 0:
            raise ValueError('No gt or bboxes')


        # overlaps = rbox_overlaps(gt_bboxes, bboxes).t()
        overlaps = self.iou_calculator(gt_bboxes, bboxes).t()
        # print(overlaps.shape)

        # if type(gt_bboxes) is not torch.tensor:
        #     gt_bboxes = torch.Tensor(gt_bboxes).to(bboxes.device)

        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             0,
                                             dtype=torch.long)
        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                # assigned_labels = overlaps.new_full((num_bboxes, ),
                #                                     -1,
                #                                     dtype=torch.long)
                assigned_labels = overlaps.new_zeros((num_bboxes, ),
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        # gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        # gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_cx = gt_bboxes[:, 0]
        gt_cy = gt_bboxes[:, 1]
        gt_points = torch.stack((gt_cx, gt_cy), dim=1).float()

        # bboxes_cx = (bboxes[:, 0] + bboxes[:, 2]) / 2.0
        # bboxes_cy = (bboxes[:, 1] + bboxes[:, 3]) / 2.0
        bboxes_cx = bboxes[:, 0]
        bboxes_cy = bboxes[:, 1]
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)

        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # Selecting candidates based on the center distance
        candidate_idxs = []
        # _, topk_idxs = distances.topk(self.topk, dim=0, largest=False)
        # candidate_idxs.append(topk_idxs)
        # candidate_idxs = torch.cat(candidate_idxs, dim=0)


        start_idx = 0
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            _, topk_idxs_per_level = distances_per_level.topk(
                self.topk, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold

        gt_bboxes_ratios = self.AspectRatio(gt_bboxes)
        gt_bboxes_ratios_per_gt = gt_bboxes_ratios.mean(0)

        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        iou_thr_weight = torch.exp((-1 / 4) * gt_bboxes_ratios_per_gt)

        # clamp neg min threshold
        # overlaps_thr_per_gt = overlaps_thr_per_gt.clamp_min(0.3)

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :] * iou_thr_weight

        # limit the positive sample's center in gt
        from mmdet.ops.point_justify import pointsJf

        # inside_flag = torch.full([num_bboxes, num_gt], 0.).to(gt_bboxes.device).float()
        inside_flag = torch.full([num_bboxes, num_gt], 0.).to(gt_bboxes.device).to(torch.float32)
        # inside_flag = torch.full([num_bboxes, num_gt], 0.).to(gt_bboxes.device).to(torch.float64)
        # print('inside_flag', inside_flag.shape, inside_flag.dtype)
        # print('gt_bboxes', gt_bboxes.shape, gt_bboxes.to(torch.float32).dtype)
        # print('rbox2poly_torch', rbox2poly_torch(gt_bboxes).shape, rbox2poly_torch(gt_bboxes).to(torch.float32).dtype)
        pointsJf(bboxes_points, \
                 rotated_box_to_poly(gt_bboxes.to(torch.float32)).contiguous().to(torch.float32), \
                 inside_flag)
        # print('inside_flag', inside_flag, torch.where(inside_flag>0))
        is_in_gts = inside_flag[candidate_idxs, torch.arange(num_gt)].to(is_pos.dtype)

        is_pos = is_pos & is_in_gts
        # print('is_pos', is_pos)
        '''
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        ep_bboxes_cy = bboxes_cy.view(1, -1).expand(
            num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)

        # calculate the left, top, right, bottom distance between positive
        # bbox center and gt side
        l_ = ep_bboxes_cx[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 0]
        t_ = ep_bboxes_cy[candidate_idxs].view(-1, num_gt) - gt_bboxes[:, 1]
        r_ = gt_bboxes[:, 2] - ep_bboxes_cx[candidate_idxs].view(-1, num_gt)
        b_ = gt_bboxes[:, 3] - ep_bboxes_cy[candidate_idxs].view(-1, num_gt)
        is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 0.01
        is_pos = is_pos & is_in_gts
        '''
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        candidate_idxs = candidate_idxs.view(-1)

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            # pos_inds = torch.nonzero(
            #     assigned_gt_inds > 0, as_tuple=False).squeeze()
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)



    def AspectRatio(self, gt_rbboxes):
        # gt_rbboxes = torch.squeeze(gt_rbboxes)
        # print('AspectRatio.gt_rbboxes')
        # print(gt_rbboxes.size())
        # gt_rbboxes = rotated_box_to_poly(gt_rbboxes.to(torch.float32)).contiguous().to(torch.float64)

        # pt1, pt2, pt3, pt4 = gt_rbboxes[..., :8].chunk(4, 1)
        #
        # edge1 = torch.sqrt(
        #     torch.pow(pt1[..., 0] - pt2[..., 0], 2) + torch.pow(pt1[..., 1] - pt2[..., 1], 2))
        # edge2 = torch.sqrt(
        #     torch.pow(pt2[..., 0] - pt3[..., 0], 2) + torch.pow(pt2[..., 1] - pt3[..., 1], 2))
        edge1 = gt_rbboxes[..., 2]
        edge2 = gt_rbboxes[..., 3]
        edges = torch.stack([edge1, edge2], dim=1)

        width, _ = torch.max(edges, 1)
        height, _ = torch.min(edges, 1)

        ratios = (width / height)
        return ratios

