"""
  The code of this file is based on https://github.com/CAPTAIN-WHU/DOTA_devkit/blob/master/ResultMerge.py, some functions are changed
  for the evaluation of icadr2015 dataset.
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, '..')
import DOTA_devkit.ucasaod_utils as util
import DOTA_devkit.polyiou.polyiou as polyiou
import pdb
import math
from multiprocessing import Pool
from functools import partial

## the thresh for nms when merge image
nms_thresh = 0.01


def py_cpu_nms_poly(dets, thresh):
    scores = dets[:, 8]
    polys = []
    areas = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                           dets[i][2], dets[i][3],
                                           dets[i][4], dets[i][5],
                                           dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        for j in range(order.size - 1):
            iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
            ovr.append(iou)
        ovr = np.array(ovr)

        # print('ovr: ', ovr)
        # print('thresh: ', thresh)
        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(ovr <= thresh)[0]
        # print('inds: ', inds)

        order = order[inds + 1]

    return keep


def py_cpu_nms_poly_fast(dets, thresh):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                           dets[i][2], dets[i][3],
                                           dets[i][4], dets[i][5],
                                           dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        # if order.size == 0:
        #     break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        # h_keep_inds = np.where(hbb_ovr == 0)[0]
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou
            # ovr.append(iou)
            # ovr_index.append(tmp_order[j])

        # ovr = np.array(ovr)
        # ovr_index = np.array(ovr_index)
        # print('ovr: ', ovr)
        # print('thresh: ', thresh)
        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]

        # order_obb = ovr_index[inds]
        # print('inds: ', inds)
        # order_hbb = order[h_keep_inds + 1]
        order = order[inds + 1]
        # pdb.set_trace()
        # order = np.concatenate((order_obb, order_hbb), axis=0).astype(np.int)
    return keep


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    # print('dets:', dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ## index for dets
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def nmsbynamedict(nameboxdict, nms, thresh):
    nameboxnmsdict = {x: [] for x in nameboxdict}
    for imgname in nameboxdict:
        # print('imgname:', imgname)
        # keep = py_cpu_nms(np.array(nameboxdict[imgname]), thresh)
        # print('type nameboxdict:', type(nameboxnmsdict))
        # print('type imgname:', type(imgname))
        # print('type nms:', type(nms))
        keep = nms(np.array(nameboxdict[imgname]), thresh)
        # print('keep:', keep)
        outdets = []
        # print('nameboxdict[imgname]: ', nameboxnmsdict[imgname])
        for index in keep:
            # print('index:', index)
            outdets.append(nameboxdict[imgname][index])
        nameboxnmsdict[imgname] = outdets
    return nameboxnmsdict


def poly2origpoly(poly, x, y, rate):
    origpoly = []
    for i in range(int(len(poly) / 2)):
        tmp_x = float(poly[i * 2] + x) / float(rate)
        tmp_y = float(poly[i * 2 + 1] + y) / float(rate)
        origpoly.append(tmp_x)
        origpoly.append(tmp_y)
    return origpoly

def order_points_new(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    if leftMost[0, 1] != leftMost[1, 1]:
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]

    else:
        if leftMost[0, 1] > rightMost[0, 1]:
            leftMost = leftMost[np.argsort(leftMost[:, 0])[:: 1], :]
        else:
            leftMost = leftMost[np.argsort(leftMost[:, 0])[:: -1], :]
        print('leftMost', leftMost)
    (tl, bl) = leftMost
    if rightMost[0, 1] != rightMost[1, 1]:
        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    else:
        if leftMost[0, 1] > rightMost[0, 1]:
            rightMost = rightMost[np.argsort(rightMost[:, 0])[::1], :]
        else:
            rightMost = rightMost[np.argsort(rightMost[:, 0])[::-1], :]
        print('rightMost', rightMost)
    (tr, br) = rightMost
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")

def mergesingle_imgtxt(dstpath, nms, fullname):
    name = util.custombasename(fullname)
    # print('name:', name)
    dstname = os.path.join(dstpath, name + '.txt')
    print(dstname)
    with open(fullname, 'r') as f_in:
        nameboxdict = {}
        lines = f_in.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
        for splitline in splitlines:
            subname = splitline[0]
            splitname = subname.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            print('subname:', subname)
            x_y = re.findall(pattern1, subname)
            print('oriname:', oriname)
            print(x_y)
            # x_y_2 = re.findall(r'\d+', x_y[0])
            # x, y = int(x_y_2[0]), int(x_y_2[1])
            #
            # pattern2 = re.compile(r'__([\d+\.]+)__\d+___')
            #
            # rate = re.findall(pattern2, subname)[0]
            #

            confidence = splitline[1]
            poly = list(map(float, splitline[2:]))
            print(poly)
            # origpoly = poly2origpoly(poly, x, y, 1.0)
            det = poly
            det.append(confidence)
            det = list(map(float, det))
            print(det)
            if (oriname not in nameboxdict):
                nameboxdict[oriname] = []
            nameboxdict[oriname].append(det)
        nameboxnmsdict = nmsbynamedict(nameboxdict, nms, nms_thresh)
        for imgname in nameboxnmsdict:
            with open(os.path.join(dstpath, 'res_' + imgname + '.txt'), 'a+') as f_out:
            # for imgname in nameboxnmsdict:
                for det in nameboxnmsdict[imgname]:
                    # print('det:', det)
                    confidence = det[-1]
                    bbox = det[0:-1]
                    # if confidence > 0.3:
                    p = np.array([[int(bbox[-8]), int(bbox[-7])], [int(bbox[-6]), int(bbox[-5])],
                                      [int(bbox[-4]), int(bbox[-3])], [int(bbox[-2]), int(bbox[-1])]])
                    p = order_points_new(p)
                    p = p.astype(np.int32)

                    f_out.write(
                        str(p[0][0]) + ',' + str(p[0][1]) + ',' + str(p[1][0]) + ',' + str(
                            p[1][1]) + ',' +
                        str(p[2][0]) + ',' + str(p[2][1]) + ',' + str(p[3][0]) + ',' + str(
                            p[3][1]) + ',' + str(det[-1]) + '\n')

                    # outline = str(confidence) + ' ' + ' '.join(map(str, bbox))
                    # # print('outline:', outline)
                    # f_out.write(outline + '\n')


def mergebase_parallel(srcpath, dstpath, nms):
    pool = Pool(16)
    filelist = util.GetFileFromThisRootDir(srcpath)

    mergesingle_fn = partial(mergesingle_imgtxt, dstpath, nms)
    # pdb.set_trace()
    pool.map(mergesingle_fn, filelist)


def mergebase(srcpath, dstpath, nms):
    filelist = util.GetFileFromThisRootDir(srcpath)
    for filename in filelist:
        mergesingle_imgtxt(dstpath, nms, filename)


def mergebyrec(srcpath, dstpath):

    mergebase(srcpath,
              dstpath,
              py_cpu_nms)


def mergebypoly(srcpath, dstpath):
    mergebase_parallel(srcpath,
                       dstpath,
                       py_cpu_nms_poly_fast)


if __name__ == '__main__':
    # mergebyrec(r'work_dirs/temp/result_raw', r'work_dirs/temp/result_task2')
    py_cpu_nms_poly_fast(r'work_dirs/temp/result_raw', )
# mergebyrec()
