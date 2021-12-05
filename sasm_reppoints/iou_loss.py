import torch
import torch.nn as nn

from mmdet.core import bbox_overlaps
from ..registry import LOSSES
from .utils import weighted_loss

from torch.autograd import Function
from torch.autograd.function import once_differentiable
from mmdet.ops.iou import convex_giou

@weighted_loss
def iou_loss(pred, target, eps=1e-6):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    loss = -ious.log()
    return loss


@weighted_loss
def bounded_iou_loss(pred, target, beta=0.2, eps=1e-3):
    """Improving Object Localization with Fitness NMS and Bounded IoU Loss,
    https://arxiv.org/abs/1711.00164.

    Args:
        pred (tensor): Predicted bboxes.
        target (tensor): Target bboxes.
        beta (float): beta parameter in smoothl1.
        eps (float): eps to avoid NaN.
    """
    pred_ctrx = (pred[:, 0] + pred[:, 2]) * 0.5
    pred_ctry = (pred[:, 1] + pred[:, 3]) * 0.5
    pred_w = pred[:, 2] - pred[:, 0] + 1
    pred_h = pred[:, 3] - pred[:, 1] + 1
    with torch.no_grad():
        target_ctrx = (target[:, 0] + target[:, 2]) * 0.5
        target_ctry = (target[:, 1] + target[:, 3]) * 0.5
        target_w = target[:, 2] - target[:, 0] + 1
        target_h = target[:, 3] - target[:, 1] + 1

    dx = target_ctrx - pred_ctrx
    dy = target_ctry - pred_ctry

    loss_dx = 1 - torch.max(
        (target_w - 2 * dx.abs()) /
        (target_w + 2 * dx.abs() + eps), torch.zeros_like(dx))
    loss_dy = 1 - torch.max(
        (target_h - 2 * dy.abs()) /
        (target_h + 2 * dy.abs() + eps), torch.zeros_like(dy))
    loss_dw = 1 - torch.min(target_w / (pred_w + eps), pred_w /
                            (target_w + eps))
    loss_dh = 1 - torch.min(target_h / (pred_h + eps), pred_h /
                            (target_h + eps))
    loss_comb = torch.stack([loss_dx, loss_dy, loss_dw, loss_dh],
                            dim=-1).view(loss_dx.size(0), -1)

    loss = torch.where(loss_comb < beta, 0.5 * loss_comb * loss_comb / beta,
                       loss_comb - 0.5 * beta)
    return loss


@weighted_loss
def giou_loss(pred, target, eps=1e-7):
    """
    Generalized Intersection over Union: A Metric and A Loss for
    Bounding Box Regression
    https://arxiv.org/abs/1902.09630

    code refer to:
    https://github.com/sfzhang15/ATSS/blob/master/atss_core/modeling/rpn/atss/loss.py#L36

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    # overlap
    lt = torch.max(pred[:, :2], target[:, :2])
    rb = torch.min(pred[:, 2:], target[:, 2:])
    wh = (rb - lt + 1).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]

    # union
    ap = (pred[:, 2] - pred[:, 0] + 1) * (pred[:, 3] - pred[:, 1] + 1)
    ag = (target[:, 2] - target[:, 0] + 1) * (target[:, 3] - target[:, 1] + 1)
    union = ap + ag - overlap + eps

    # IoU
    ious = overlap / union

    # enclose area
    enclose_x1y1 = torch.min(pred[:, :2], target[:, :2])
    enclose_x2y2 = torch.max(pred[:, 2:], target[:, 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1 + 1).clamp(min=0)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1] + eps

    # GIoU
    gious = ious - (enclose_area - union) / enclose_area
    loss = 1 - gious
    return loss


@LOSSES.register_module
class IoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(IoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * iou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module
class BoundedIoULoss(nn.Module):

    def __init__(self, beta=0.2, eps=1e-3, reduction='mean', loss_weight=1.0):
        super(BoundedIoULoss, self).__init__()
        self.beta = beta
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * bounded_iou_loss(
            pred,
            target,
            weight,
            beta=self.beta,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module
class GIoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * giou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


class ConvexGIoULossFuction(Function):

    @staticmethod
    def forward(ctx, pred, target, weight=None, reduction=None, avg_factor=None, loss_weight=1.0):
        ctx.save_for_backward(pred)

        convex_gious, grad = convex_giou(pred, target)


        loss = (1 - convex_gious)


        if weight is not None:
            loss = loss * weight
            grad = grad * weight.reshape(-1, 1)
        if reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'mean':
            loss = loss.mean()
        else:
            loss = loss

        # _unvalid_grad_filter
        eps = 1e-6
        unvaild_inds = torch.nonzero((grad > 1).sum(1))[:, 0]
        grad[unvaild_inds] = eps

        # _reduce_grad
        reduce_grad = -grad / grad.size(0) * loss_weight
        ctx.convex_points_grad = reduce_grad
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, input=None):
        convex_points_grad = ctx.convex_points_grad
        return convex_points_grad, None, None, None, None, None


convex_giou_loss = ConvexGIoULossFuction.apply


@LOSSES.register_module
class ConvexGIoULoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(ConvexGIoULoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight.unsqueeze(-1)).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * convex_giou_loss(
            pred,
            target,
            weight,
            reduction,
            avg_factor,
            self.loss_weight)
        return loss


class BCConvexGIoULossFuction(Function):

    @staticmethod
    def forward(ctx, pred, target, weight=None, reduction=None, avg_factor=None, loss_weight=1.0):
        ctx.save_for_backward(pred)

        # print(pred.size())

        convex_gious, grad = convex_giou(pred, target)
        # print(target.size())

        pts_pred_all_dx = pred[:, 0::2]
        pts_pred_all_dy = pred[:, 1::2]

        pred_left_x_inds = pts_pred_all_dx.min(dim=1, keepdim=True)[1]
        pred_right_x_inds = pts_pred_all_dx.max(dim=1, keepdim=True)[1]
        pred_up_y_inds = pts_pred_all_dy.min(dim=1, keepdim=True)[1]
        pred_bottom_y_inds = pts_pred_all_dy.max(dim=1, keepdim=True)[1]

        pred_right_x = pts_pred_all_dx.gather(dim=1, index=pred_right_x_inds)
        pred_right_y = pts_pred_all_dy.gather(dim=1, index=pred_right_x_inds)

        pred_left_x = pts_pred_all_dx.gather(dim=1, index=pred_left_x_inds)
        pred_left_y = pts_pred_all_dy.gather(dim=1, index=pred_left_x_inds)

        pred_up_x = pts_pred_all_dx.gather(dim=1, index=pred_up_y_inds)
        pred_up_y = pts_pred_all_dy.gather(dim=1, index=pred_up_y_inds)

        pred_bottom_x = pts_pred_all_dx.gather(dim=1, index=pred_bottom_y_inds)
        pred_bottom_y = pts_pred_all_dy.gather(dim=1, index=pred_bottom_y_inds)
        pred_corners = torch.cat([pred_left_x, pred_left_y,
            pred_up_x, pred_up_y, pred_right_x, pred_right_y, pred_bottom_x, pred_bottom_y], dim=-1)

        pts_target_all_dx = target[:, 0::2]
        pts_target_all_dy = target[:, 1::2]

        target_left_x_inds = pts_target_all_dx.min(dim=1, keepdim=True)[1]
        target_right_x_inds = pts_target_all_dx.max(dim=1, keepdim=True)[1]
        target_up_y_inds = pts_target_all_dy.min(dim=1, keepdim=True)[1]
        target_bottom_y_inds = pts_target_all_dy.max(dim=1, keepdim=True)[1]

        target_right_x = pts_target_all_dx.gather(dim=1, index=target_right_x_inds)
        target_right_y = pts_target_all_dy.gather(dim=1, index=target_right_x_inds)

        target_left_x = pts_target_all_dx.gather(dim=1, index=target_left_x_inds)
        target_left_y = pts_target_all_dy.gather(dim=1, index=target_left_x_inds)

        target_up_x = pts_target_all_dx.gather(dim=1, index=target_up_y_inds)
        target_up_y = pts_target_all_dy.gather(dim=1, index=target_up_y_inds)

        target_bottom_x = pts_target_all_dx.gather(dim=1, index=target_bottom_y_inds)
        target_bottom_y = pts_target_all_dy.gather(dim=1, index=target_bottom_y_inds)

        target_corners = torch.cat([target_left_x, target_left_y,
                                  target_up_x, target_up_y, target_right_x, target_right_y, target_bottom_x, target_bottom_y],
                                 dim=-1)
        # print(pred_corners, target_corners)



        pts_pred_dx_mean = pts_pred_all_dx.mean(dim=1, keepdim=True).reshape(-1, 1)
        pts_pred_dy_mean = pts_pred_all_dy.mean(dim=1, keepdim=True).reshape(-1, 1)
        pts_pred_mean = torch.cat([pts_pred_dx_mean, pts_pred_dy_mean], dim=-1)

        pts_target_dx_mean = pts_target_all_dx.mean(dim=1, keepdim=True).reshape(-1, 1)
        pts_target_dy_mean = pts_target_all_dy.mean(dim=1, keepdim=True).reshape(-1, 1)
        pts_target_mean = torch.cat([pts_target_dx_mean, pts_target_dy_mean], dim=-1)



        # diff = torch.pow(torch.pow(pts_target_dy_mean - pts_pred_dy_mean, 2)
        #                 + torch.pow(pts_target_dx_mean - pts_pred_dx_mean, 2), 1 / 2).reshape(-1, 1)
        #
        # iou_weight = (torch.pow(torch.log((dis * dis) + 1), 2) + 1)
        #
        # target_dis = torch.pow(torch.pow(target[:, 0] - target[:, 2], 2)
        #                 + torch.pow(target[:, 1] - target[:, 3], 2), 1 / 2).reshape(-1, 1)


        beta = 1.0

        # 中心点
        diff_mean = torch.abs(pts_pred_mean - pts_target_mean)
        diff_mean_loss = torch.where(diff_mean < beta, 0.5 * diff_mean * diff_mean / beta,
                                diff_mean - 0.5 * beta)
        diff_mean_loss = diff_mean_loss.sum() / len(diff_mean_loss)

        # 角点差
        diff_corners = torch.abs(pred_corners - target_corners)
        diff_corners_loss = torch.where(diff_corners < beta, 0.5 * diff_corners * diff_corners / beta,
                                diff_corners - 0.5 * beta)
        diff_corners_loss = diff_corners_loss.sum() / len(diff_corners_loss)


        target_aspect = AspectRatio(target)
        # the key is how to find the function of dynamic assign weight
        smooth_loss_weight = torch.exp((-1 / 4) * target_aspect)
        # 09_10
        loss = smooth_loss_weight * (diff_mean_loss.reshape(-1, 1).cuda() +
                                     diff_corners_loss.reshape(-1, 1).cuda()) \
                                    + 1 - (1 - 2 * smooth_loss_weight) * convex_gious


        # loss = diff_mean_loss.reshape(-1, 1).cuda() + diff_corners_loss.reshape(-1, 1).cuda() + 1 - convex_gious
        if weight is not None:
            loss = loss * weight
            grad = grad * weight.reshape(-1, 1)
        if reduction == 'sum':
            loss = loss.sum()
        elif reduction == 'mean':
            loss = loss.mean()
        else:
            loss = loss

        # _unvalid_grad_filter
        eps = 1e-6
        unvaild_inds = torch.nonzero((grad > 1).sum(1))[:, 0]
        grad[unvaild_inds] = eps

        # _reduce_grad
        reduce_grad = -grad / grad.size(0) * loss_weight
        ctx.convex_points_grad = reduce_grad
        return loss
    @staticmethod
    @once_differentiable
    def backward(ctx, input=None):
        convex_points_grad = ctx.convex_points_grad
        return convex_points_grad, None, None, None, None, None


bc_convex_giou_loss = BCConvexGIoULossFuction.apply


@LOSSES.register_module
class BCConvexGIoULoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(BCConvexGIoULoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            return (pred * weight.unsqueeze(-1)).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * bc_convex_giou_loss(
            pred,
            target,
            weight,
            reduction,
            avg_factor,
            self.loss_weight)
        return loss

def AspectRatio(gt_rbboxes):
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