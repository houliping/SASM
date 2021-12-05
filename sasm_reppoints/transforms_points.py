import math

import numpy as np
import torch


def norm_angle(angle, range=[-np.pi / 4, np.pi]):
    return (angle - range[0]) % range[1] + range[0]

def AspectRatio(rbboxes):
    # gt_rbboxes = torch.squeeze(gt_rbboxes)
    # print('AspectRatio.gt_rbboxes')
    # print(gt_rbboxes.size())

    '''

    rbboxes:[x0,y0,x1,y1,x2,y2,x3,y3]
    Returns: the aspect ratio of rbbox

    '''

    pt1, pt2, pt3, pt4 = rbboxes[..., :8].chunk(4, 1)

    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) + torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) + torch.pow(pt2[..., 1] - pt3[..., 1], 2))

    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    ratios = (width / height)
    return ratios


def points_center_pts(RPoints, y_first=True):

    '''
    RPoints:[:, 18]  the  lists of Pointsets (9 points)
    points_center_pts: the mean_center coordination of Pointsets

    '''
    RPoints = RPoints.reshape(-1, 9, 2)

    if y_first:
        pts_dy = RPoints[:, :, 0::2]
        pts_dx = RPoints[:, :, 1::2]
    else:
        pts_dx = RPoints[:, :, 0::2]
        pts_dy = RPoints[:, :, 1::2]
    pts_dy_mean = pts_dy.mean(dim=1, keepdim=True).reshape(-1, 1)
    pts_dx_mean = pts_dx.mean(dim=1, keepdim=True).reshape(-1, 1)
    points_center_pts = torch.cat([pts_dx_mean, pts_dy_mean], dim=1).reshape(-1, 2)
    return points_center_pts


def rbboxes_center_pts(rbboxes):
    """
    rbboxes:n*8
    rbboxes:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rbboxes_center_pts:[x_ctr,y_ctr]
    """
    pt1, pt2, pt3, pt4 = rbboxes[..., :8].chunk(4, 1)

    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) + torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) + torch.pow(pt2[..., 1] - pt3[..., 1], 2))

    angles1 = torch.atan2((pt2[..., 1] - pt1[..., 1]), (pt2[..., 0] - pt1[..., 0]))
    angles2 = torch.atan2((pt4[..., 1] - pt1[..., 1]), (pt4[..., 0] - pt1[..., 0]))
    angles = rbboxes.new_zeros(rbboxes.shape[0])
    angles[edge1 > edge2] = angles1[edge1 > edge2]
    angles[edge1 <= edge2] = angles2[edge1 <= edge2]

    angles = norm_angle(angles)

    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0

    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    xy_ctr = torch.stack([x_ctr, y_ctr], 1)

    return xy_ctr, width, height, angles




