import torch

from ..bbox import PseudoSampler, assign_and_sample, build_assigner, build_bbox_coder
from ..utils import multi_apply


def sam_anchor_target(anchor_list,
                  valid_flag_list,
                  gt_bboxes_list,
                  img_metas,
                  target_means,
                  target_stds,
                  cfg,
                  gt_bboxes_ignore_list=None,
                  gt_labels_list=None,
                  label_channels=1,
                  sampling=True,
                  unmap_outputs=True):
    """Compute regression and classification targets for anchors.

    Args:
        anchor_list (list[list]): Multi level anchors of each image.
        valid_flag_list (list[list]): Multi level valid flags of each image.
        gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
        img_metas (list[dict]): Meta info of each image.
        target_means (Iterable): Mean value of regression targets.
        target_stds (Iterable): Std value of regression targets.
        cfg (dict): RPN train configs.

    Returns:
        tuple
    """
    num_imgs = len(img_metas)
    assert len(anchor_list) == len(valid_flag_list) == num_imgs

    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]

    # anchor number of multi levels
    num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
    # concat all level anchors and flags to a single tensor
    for i in range(num_imgs):
        assert len(anchor_list[i]) == len(valid_flag_list[i])
        anchor_list[i] = torch.cat(anchor_list[i])
        valid_flag_list[i] = torch.cat(valid_flag_list[i])

    # compute targets for each image
    if gt_bboxes_ignore_list is None:
        gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
    if gt_labels_list is None:
        gt_labels_list = [None for _ in range(num_imgs)]


    # num_level_proposals = [anchor.size(0) for anchor in anchor_list[0]]
    num_level_proposals_list = [num_level_anchors] * num_imgs

    (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
     pos_inds_list, neg_inds_list, all_sam_weights) = multi_apply(
        sam_anchor_target_single,
        anchor_list,
        num_level_proposals_list,
        valid_flag_list,
        gt_bboxes_list,
        gt_bboxes_ignore_list,
        gt_labels_list,
        img_metas,
        target_means=target_means,
        target_stds=target_stds,
        cfg=cfg,
        label_channels=label_channels,
        sampling=sampling,
        unmap_outputs=unmap_outputs)
    # no valid anchors
    if any([labels is None for labels in all_labels]):
        return None
    # sampled anchors of all images
    num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
    num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
    # split targets to a list w.r.t. multiple levels
    labels_list = sam_images_to_levels(all_labels, num_level_anchors)
    label_weights_list = sam_images_to_levels(all_label_weights, num_level_anchors)
    bbox_targets_list = sam_images_to_levels(all_bbox_targets, num_level_anchors)
    bbox_weights_list = sam_images_to_levels(all_bbox_weights, num_level_anchors)
    sam_weights_list = sam_images_to_levels(all_bbox_weights, num_level_anchors)

    return (labels_list, label_weights_list, bbox_targets_list,
            bbox_weights_list, num_total_pos, num_total_neg, sam_weights_list)


def sam_images_to_levels(target, num_level_anchors):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_level_anchors:
        end = start + n
        level_targets.append(target[:, start:end].squeeze(0))
        start = end
    return level_targets


def sam_anchor_target_single(flat_anchors,
                         num_level_proposals,
                         valid_flags,
                         gt_bboxes,
                         gt_bboxes_ignore,
                         gt_labels,
                         img_meta,
                         target_means,
                         target_stds,
                         cfg,
                         label_channels=1,
                         sampling=True,
                         unmap_outputs=True):
    bbox_coder_cfg = cfg.get('bbox_coder', '')
    if bbox_coder_cfg == '':
        bbox_coder_cfg = dict(type='DeltaXYWHBBoxCoder')
    bbox_coder = build_bbox_coder(bbox_coder_cfg)
    # Set True to use IoULoss
    reg_decoded_bbox = cfg.get('reg_decoded_bbox', False)

    inside_flags = sam_anchor_inside_flags(flat_anchors, valid_flags,
                                       img_meta['img_shape'][:2],
                                       cfg.allowed_border)
    if not inside_flags.any():
        return (None,) * 6
    # assign gt and sample anchors
    anchors = flat_anchors[inside_flags, :]

    num_level_anchors_inside = sam_get_num_level_anchors_inside(num_level_proposals, valid_flags)

    if sampling:
        assign_result, sampling_result = assign_and_sample(
            anchors, gt_bboxes, gt_bboxes_ignore, None, cfg)
    else:
        bbox_assigner = build_assigner(cfg.assigner)
        if cfg.assigner.type !="ATSSAssigner" and cfg.assigner.type !="SASAssigner" :
            assign_result = bbox_assigner.assign(anchors, gt_bboxes,
                                                 gt_bboxes_ignore, gt_labels)
        else:
            assign_result = bbox_assigner.assign(anchors, num_level_anchors_inside, gt_bboxes,
                                             gt_bboxes_ignore, gt_labels)
        bbox_sampler = PseudoSampler()
        sampling_result = bbox_sampler.sample(assign_result, anchors,
                                              gt_bboxes)

    num_valid_anchors = anchors.shape[0]
    gt_inds = assign_result.gt_inds
    bbox_gt = anchors.new_zeros([num_valid_anchors, 5])
    bbox_targets = torch.zeros_like(anchors)
    bbox_weights = torch.zeros_like(anchors)
    labels = anchors.new_zeros(num_valid_anchors, dtype=torch.long)
    label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)





    pos_inds = sampling_result.pos_inds
    neg_inds = sampling_result.neg_inds
    if len(pos_inds) > 0:
        if not reg_decoded_bbox:
            pos_bbox_targets = bbox_coder.encode(sampling_result.pos_bboxes,
                                                sampling_result.pos_gt_bboxes)
        else:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
        bbox_gt[pos_inds, :] = pos_bbox_targets.float()
        bbox_targets[pos_inds, :] = pos_bbox_targets.to(bbox_targets)
        bbox_weights[pos_inds, :] = 1.0
        if gt_labels is None:
            labels[pos_inds] = 1
        else:
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        if cfg.pos_weight <= 0:
            label_weights[pos_inds] = 1.0
        else:
            label_weights[pos_inds] = cfg.pos_weight
    if len(neg_inds) > 0:
        label_weights[neg_inds] = 1.0

    rbboxes_center, width, height, angles = bbox_gt[:, :2], bbox_targets[:, 2], bbox_targets[:, 3], bbox_targets[:, 4]
    anchor_center = bbox_targets[:, :2]

    distances = torch.zeros_like(angles).reshape(-1)

    angles_index_wh = (width != 0) & (angles >= 0) & (angles <= 1.57)
    angles_index_hw = (width != 0) & ((angles < 0) | (angles > 1.57))

    # 01_la:compution of distance
    distances[angles_index_wh] = torch.sqrt(
        (torch.pow(rbboxes_center[angles_index_wh, 0] - anchor_center[angles_index_wh, 0], 2) / width[angles_index_wh])
        + (torch.pow(rbboxes_center[angles_index_wh, 1] - anchor_center[angles_index_wh, 1], 2) / height[
            angles_index_wh]))


    distances[angles_index_hw] = torch.sqrt(
        (torch.pow(rbboxes_center[angles_index_hw, 0] - anchor_center[angles_index_hw, 0], 2) / height[angles_index_hw])
        + (torch.pow(rbboxes_center[angles_index_hw, 1] - anchor_center[angles_index_hw, 1], 2) / width[
            angles_index_hw]))



    distances[distances == float('nan')] = 0.

    sam_weights = label_weights * (torch.exp(1 / (distances + 1)))
    sam_weights[sam_weights == float('inf')] = 0.


    # map up to original set of anchors
    if unmap_outputs:
        num_total_anchors = flat_anchors.size(0)
        labels = sam_unmap(labels, num_total_anchors, inside_flags)
        label_weights = sam_unmap(label_weights, num_total_anchors, inside_flags)
        bbox_targets = sam_unmap(bbox_targets, num_total_anchors, inside_flags)
        bbox_weights = sam_unmap(bbox_weights, num_total_anchors, inside_flags)
        sam_weights = sam_unmap(sam_weights, num_total_anchors, inside_flags)

    return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
            neg_inds, sam_weights)


# TODO for rotated box
def sam_anchor_inside_flags(flat_anchors, valid_flags, img_shape,
                        allowed_border=0):
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
                       (flat_anchors[:, 0] >= -allowed_border).type(torch.uint8) & \
                       (flat_anchors[:, 1] >= -allowed_border).type(torch.uint8) & \
                       (flat_anchors[:, 2] < img_w + allowed_border).type(torch.uint8) & \
                       (flat_anchors[:, 3] < img_h + allowed_border).type(torch.uint8)
    else:
        inside_flags = valid_flags
    return inside_flags


def sam_unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count,), fill)
        ret[inds] = data
    else:
        new_size = (count,) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret

def sam_get_num_level_anchors_inside(num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside