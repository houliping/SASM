import torch

from ..geometry import bbox_overlaps
from .assign_result import AssignResult
from .base_assigner import BaseAssigner

from mmdet.ops.iou import convex_iou


from memory_profiler import profile


class SAMaxConvexIoUAssigner(BaseAssigner):
    def __init__(self,
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=-1):

        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        # self.pos_iou_thr_coarse = pos_iou_thr_coarse
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr

    def assign(self, points, gt_rbboxes, overlaps,
               gt_rbboxes_ignore=None, gt_labels=None, ):

        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
                gt_rbboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = points.device
            bboxes = points.cpu()
            gt_rbboxes = gt_rbboxes.cpu()
            if gt_rbboxes_ignore is not None:
                gt_rbboxes_ignore = gt_rbboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        # print('gt_rbboxes.size()')
        # print( gt_rbboxes.size())
        # print('points.size()')
        # print(points.size())

        if overlaps is None:
            overlaps = self.convex_overlaps(gt_rbboxes, points)

        if (self.ignore_iof_thr > 0 and gt_rbboxes_ignore is not None
                and gt_rbboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.convex_overlaps(
                    bboxes, gt_rbboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.convex_overlaps(
                    gt_rbboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        # print('gt_rbboxes222222.size()')
        # print(gt_rbboxes.size())

        # print('overlaps.size()')
        # print(overlaps.size())

        assign_result = self.assign_wrt_overlaps(overlaps, gt_rbboxes, gt_labels)

        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result

    def assign_wrt_overlaps(self, overlaps, gt_rbboxes, gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_zeros((num_bboxes,),
                                                     dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        # 最大值及最大值索引
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)


        ratios = self.AspectRatio(gt_rbboxes[argmax_overlaps])


        one = torch.ones_like(ratios)

        iou_thr_weight = torch.exp((-1 / 4) * ratios)




        # 2. assign negative: below

        neg_iou_thr_refine = self.neg_iou_thr

        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < neg_iou_thr_refine)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= neg_iou_thr_refine[0])
                             & (max_overlaps < neg_iou_thr_refine[1])] = 0


        # 3. assign positive: above positive IoU threshold
        # pos_inds_coarse = max_overlaps >= self.pos_iou_thr_coarse


        pos_iou_thr_refine = self.pos_iou_thr * iou_thr_weight

        pos_inds = max_overlaps >= pos_iou_thr_refine

        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. assign fg: for each gt, proposals with highest IoU
        # print(num_gts)
        for i in range(num_gts):
            if gt_max_overlaps[i] >= self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                    assigned_gt_inds[max_iou_inds] = i + 1
                else:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_zeros((num_bboxes,))
            pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)




    def points_center_pts(self, pts, y_first=True):

        if y_first:
            pts = pts.reshape(-1, 9, 2)
            pts_dy = pts[:, :, 0::2]
            pts_dx = pts[:, :, 1::2]
            pts_dy_mean = pts_dy.mean(dim=1, keepdim=True).reshape(-1, 1)
            pts_dx_mean = pts_dx.mean(dim=1, keepdim=True).reshape(-1, 1)
            #RPoints = torch.cat([pts_dx_mean, pts_dy_mean], dim=2).reshape(-1, 2)
            dis = torch.pow(torch.pow(pts_dy_mean, 2) + torch.pow(pts_dx_mean, 2), 1/2).reshape(-1, 1)
        else:
            pts = pts.reshape(-1, 9, 2)
            pts_dx = pts[:, :, 0::2]
            pts_dy = pts[:, :, 1::2]
            pts_dy_mean = pts_dy.mean(dim=1, keepdim=True).reshape(-1, 1)
            pts_dx_mean = pts_dx.mean(dim=1, keepdim=True).reshape(-1, 1)
            #RPoints = torch.cat([pts_dx_mean, pts_dy_mean], dim=2).reshape(-1, 2)
            dis = torch.pow(torch.pow(pts_dy_mean, 2) + torch.pow(pts_dx_mean, 2), 1/2).reshape(-1, 1)
        return dis

    def convex_overlaps(self, gt_rbboxes, points):
        overlaps = convex_iou(points, gt_rbboxes)
        overlaps = overlaps.transpose(1, 0)
        return overlaps

    def AspectRatio(self, gt_rbboxes):

        # gt_rbboxes = torch.squeeze(gt_rbboxes)

        # print('AspectRatio.gt_rbboxes')
        # print(gt_rbboxes.size())

        pt1, pt2, pt3, pt4 = gt_rbboxes[..., :8].chunk(4, 1)

        edge1 = torch.sqrt(
            torch.pow(pt1[..., 0] - pt2[..., 0], 2) + torch.pow(pt1[..., 1] - pt2[..., 1], 2))
        edge2 = torch.sqrt(
            torch.pow(pt2[..., 0] - pt3[..., 0], 2) + torch.pow(pt2[..., 1] - pt3[..., 1], 2))

        edges = torch.stack([edge1, edge2], dim=1)


        width, _ = torch.max(edges, 1)
        height, _ = torch.min(edges, 1)

        ratios = (width / height)
        return ratios


