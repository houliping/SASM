import os.path as osp

import mmcv
import numpy as np

# from DOTA_devkit.ResultMerge_multi_process import mergebypoly
from DOTA_devkit.icdar_nms import mergebypoly
from DOTA_devkit.flaw_evaluation_task1 import voc_eval
from mmdet.core import rotated_box_to_poly_single
from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class ICDAR2015(CustomDataset):
    CLASSES = ('text', )
    def evaluate(self, results, work_dir=None, gt_dir=None, imagesetfile=None):
        dst_raw_path = osp.join(work_dir, 'results_before_nms')
        dst_merge_path = osp.join(work_dir, 'results_after_nms')
        mmcv.mkdir_or_exist(dst_raw_path)
        mmcv.mkdir_or_exist(dst_merge_path)

        print('Saving results to {}'.format(dst_raw_path))


        self.result_to_txt(results, dst_raw_path)

        print('Merge results to {}'.format(dst_merge_path))
        mergebypoly(dst_raw_path, dst_merge_path)

        print('Get img_txt')
       

    def result_to_txt(self, results, results_path):
        img_names = [img_info['filename'] for img_info in self.img_infos]

        assert len(results) == len(img_names), 'len(results) != len(img_names)'

        for classname in self.CLASSES:
            f_out = open(osp.join(results_path, classname + '.txt'), 'w')
            print(classname + '.txt')
            # per result represent one image
            for img_id, result in enumerate(results):

                for class_id, bboxes in enumerate(result):
                    # print(result)
                    # print(class_id)
                    # print(self.CLASSES[class_id])
                    # if self.CLASSES[class_id] != classname:
                    #     continue
                    if bboxes.size != 0:
                        for bbox in bboxes:
                            score = bbox[5]
                            bbox = rotated_box_to_poly_single(bbox[:5])
                            # print(bbox)
                            temp_txt = '{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n'.format(
                                osp.splitext(img_names[img_id])[0], score, bbox[0], bbox[1], bbox[2], bbox[3], bbox[4],
                                bbox[5], bbox[6], bbox[7])
                            f_out.write(temp_txt)
            f_out.close()

