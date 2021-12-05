# -*- coding: utf-8 -*-
"""
"""

import pickle
import cv2
import json
import os 
import shutil

root = 'BeyondBoundingBox/'
detection_pkl_path = os.path.join(root, 'output/dota/02_15_atss/bbox_predict.pkl')
val_json = os.path.join(root, 'data/cfa_dota_obb_1024/test_split/test_coco.json')

result_raw_outpath = './txt_out/result_raw/'

if os.path.exists(result_raw_outpath):
    shutil.rmtree(result_raw_outpath)  # delete output folder
os.makedirs(result_raw_outpath)

#open result pkl
with open(detection_pkl_path, 'rb') as file:
    while True:
        try:
            data = pickle.load(file)
        except EOFError:
            break

#print(data_list1)
#
num_img = len(data)
#
#
for iter_img in range(num_img):
    # print(iter_img)
    img_results = data[iter_img]
    # contour_result = data[iter_img][1]
    # print('img_results', img_results)
    # print('contour_result', contour_result)
    
    #open json 
    with open(val_json) as f:
        ann=json.load(f)

        for img_item in ann['images']:
            # print(img_item['id'])
            if iter_img + 1 == img_item['id']:
                img_name = img_item['file_name']
                # print(img_name)
                img_base_name = img_name.split('.png')[0]
                # print(img_base_name)
                
                
                bboxes = img_results
                num_bboxes = len(bboxes)
    
                for iter_box in range(num_bboxes):
                    print(iter_box)

                    if len(bboxes[iter_box])>0:
                        print(bboxes[iter_box])
                        if iter_box == 0:
                            class_name = 'plane'
                        elif iter_box == 1:
                            class_name = 'baseball-diamond'
                        elif iter_box == 2:
                            class_name = 'bridge'
                        elif iter_box == 3:
                            class_name = 'ground-track-field'                
                        elif iter_box == 4:
                            class_name = 'small-vehicle'
                        elif iter_box == 5:
                            class_name = 'large-vehicle'
                        elif iter_box == 6:
                            class_name = 'ship'
                        elif iter_box == 7:
                            class_name = 'tennis-court'
                        elif iter_box == 8:
                            class_name = 'basketball-court'
                        elif iter_box == 9:
                            class_name = 'storage-tank'
                        elif iter_box == 10:
                            class_name = 'soccer-ball-field'
                        elif iter_box == 11:
                            class_name = 'roundabout'
                        elif iter_box == 12:
                            class_name = 'harbor'
                        elif iter_box == 13:
                            class_name = 'swimming-pool'
                        elif iter_box == 14:
                            class_name = 'helicopter'                
                        else:
                            class_name = None
                            
                            
                        for bbox in bboxes[iter_box]:
                            # print(bbox)

                            confidence = float(bbox[-1])
                            # fn_write_txt(outpath+txt_file,class_name, confidence, xmin, ymin ,xmax, ymax)  
                            if class_name =='harbor':
                                with open('txt_out/result_raw/Task1_harbor.txt', 'a+') as f:
                                    f.write(img_base_name+ ' ' + str(confidence) + ' ' + str(bbox[-9])+ ' ' + str(bbox[-8]) + ' ' + str(
                                        bbox[-7])+ ' ' + str(bbox[-6]) + ' ' + str(bbox[-5])+ ' ' + str(bbox[-4]) + ' ' + str(bbox[-3])+ ' ' + str(
                                        bbox[-2]) + '\n')
                                    
                            if class_name =='roundabout':
                                with open('txt_out/result_raw/Task1_roundabout.txt', 'a+') as f:
                                    f.write(img_base_name+ ' ' + str(confidence) + ' ' + str(bbox[-9])+ ' ' + str(bbox[-8]) + ' ' + str(
                                        bbox[-7])+ ' ' + str(bbox[-6]) + ' ' + str(bbox[-5])+ ' ' + str(bbox[-4]) + ' ' + str(bbox[-3])+ ' ' + str(
                                        bbox[-2]) + '\n')         
            
                            if class_name =='small-vehicle':
                                with open('txt_out/result_raw/Task1_small-vehicle.txt', 'a+') as f:
                                    f.write(img_base_name+ ' ' + str(confidence) + ' ' + str(bbox[-9])+ ' ' + str(bbox[-8]) + ' ' + str(
                                        bbox[-7])+ ' ' + str(bbox[-6]) + ' ' +  str(bbox[-5])+ ' ' + str(bbox[-4]) + ' ' + str(bbox[-3])+ ' ' + str(
                                        bbox[-2]) + '\n')  
            
            
                            if class_name =='tennis-court':
                                with open('txt_out/result_raw/Task1_tennis-court.txt', 'a+') as f:
                                    f.write(img_base_name+ ' ' + str(confidence) + ' ' + str(bbox[-9])+ ' ' + str(bbox[-8]) + ' ' + str(
                                        bbox[-7])+ ' ' + str(bbox[-6]) + ' ' +  str(bbox[-5])+ ' ' + str(bbox[-4]) + ' ' + str(bbox[-3])+ ' ' + str(
                                        bbox[-2]) + '\n')                       
            
            
                            if class_name =='baseball-diamond':
                                with open('txt_out/result_raw/Task1_baseball-diamond.txt', 'a+') as f:
                                    f.write(img_base_name+ ' ' + str(confidence) + ' ' + str(bbox[-9])+ ' ' + str(bbox[-8]) + ' ' + str(
                                        bbox[-7])+ ' ' + str(bbox[-6]) + ' ' +  str(bbox[-5])+ ' ' + str(bbox[-4]) + ' ' + str(bbox[-3])+ ' ' + str(
                                        bbox[-2]) + '\n')   
            
                            if class_name =='ship':
                                with open('txt_out/result_raw/Task1_ship.txt', 'a+') as f:
                                    f.write(img_base_name+ ' ' + str(confidence) + ' ' + str(bbox[-9])+ ' ' + str(bbox[-8]) + ' ' + str(
                                        bbox[-7])+ ' ' + str(bbox[-6]) + ' ' +  str(bbox[-5])+ ' ' + str(bbox[-4]) + ' ' + str(bbox[-3])+ ' ' + str(
                                        bbox[-2]) + '\n')  
            
                            if class_name =='large-vehicle':
                                with open('txt_out/result_raw/Task1_large-vehicle.txt', 'a+') as f:
                                    f.write(img_base_name+ ' ' + str(confidence) + ' ' + str(bbox[-9])+ ' ' + str(bbox[-8]) + ' ' + str(
                                        bbox[-7])+ ' ' + str(bbox[-6]) + ' ' +  str(bbox[-5])+ ' ' + str(bbox[-4]) + ' ' + str(bbox[-3])+ ' ' + str(
                                        bbox[-2]) + '\n')  
            
                            if class_name =='storage-tank':
                                with open('txt_out/result_raw/Task1_storage-tank.txt', 'a+') as f:
                                    f.write(img_base_name+ ' ' + str(confidence) + ' ' + str(bbox[-9])+ ' ' + str(bbox[-8]) + ' ' + str(
                                        bbox[-7])+ ' ' + str(bbox[-6]) + ' ' +  str(bbox[-5])+ ' ' + str(bbox[-4]) + ' ' + str(bbox[-3])+ ' ' + str(
                                        bbox[-2]) + '\n')  
            
            
                            if class_name =='plane':
                                with open('txt_out/result_raw/Task1_plane.txt', 'a+') as f:
                                    f.write(img_base_name+ ' ' + str(confidence) + ' ' + str(bbox[-9])+ ' ' + str(bbox[-8]) + ' ' + str(
                                        bbox[-7])+ ' ' + str(bbox[-6]) + ' ' +  str(bbox[-5])+ ' ' + str(bbox[-4]) + ' ' + str(bbox[-3])+ ' ' + str(
                                        bbox[-2]) + '\n')  
            
            
                            if class_name =='swimming-pool':
                                with open('txt_out/result_raw/Task1_swimming-pool.txt', 'a+') as f:
                                    f.write(img_base_name+ ' ' + str(confidence) + ' ' + str(bbox[-9])+ ' ' + str(bbox[-8]) + ' ' + str(
                                        bbox[-7])+ ' ' + str(bbox[-6]) + ' ' +  str(bbox[-5])+ ' ' + str(bbox[-4]) + ' ' + str(bbox[-3])+ ' ' + str(
                                        bbox[-2]) + '\n')  
            
            
                            if class_name =='bridge':
                                with open('txt_out/result_raw/Task1_bridge.txt', 'a+') as f:
                                    f.write(img_base_name+ ' ' + str(confidence) + ' ' + str(bbox[-9])+ ' ' + str(bbox[-8]) + ' ' + str(
                                        bbox[-7])+ ' ' + str(bbox[-6]) + ' ' +  str(bbox[-5])+ ' ' + str(bbox[-4]) + ' ' + str(bbox[-3])+ ' ' + str(
                                        bbox[-2]) + '\n')  
                                   
               
                            if class_name =='helicopter':
                                with open('txt_out/result_raw/Task1_helicopter.txt', 'a+') as f:
                                    f.write(img_base_name+ ' ' + str(confidence) + ' ' + str(bbox[-9])+ ' ' + str(bbox[-8]) + ' ' + str(
                                        bbox[-7])+ ' ' + str(bbox[-6]) + ' ' +  str(bbox[-5])+ ' ' + str(bbox[-4]) + ' ' + str(bbox[-3])+ ' ' + str(
                                        bbox[-2]) + '\n')  
                                 
            
                            if class_name =='ground-track-field':
                                with open('txt_out/result_raw/Task1_ground-track-field.txt', 'a+') as f:
                                    f.write(img_base_name+ ' ' + str(confidence) + ' ' + str(bbox[-9])+ ' ' + str(bbox[-8]) + ' ' + str(
                                        bbox[-7])+ ' ' + str(bbox[-6]) + ' ' +  str(bbox[-5])+ ' ' + str(bbox[-4]) + ' ' + str(bbox[-3])+ ' ' + str(
                                        bbox[-2]) + '\n')                
            
                            if class_name =='soccer-ball-field':
                                with open('txt_out/result_raw/Task1_soccer-ball-field.txt', 'a+') as f:
                                    f.write(img_base_name+ ' ' + str(confidence) + ' ' + str(bbox[-9])+ ' ' + str(bbox[-8]) + ' ' + str(
                                        bbox[-7])+ ' ' + str(bbox[-6]) + ' ' +  str(bbox[-5])+ ' ' + str(bbox[-4]) + ' ' + str(bbox[-3])+ ' ' + str(
                                        bbox[-2]) + '\n')  
                                   
                                   
                            if class_name =='basketball-court':
                                with open('txt_out/result_raw/Task1_basketball-court.txt', 'a+') as f:
                                    f.write(img_base_name+ ' ' + str(confidence) + ' ' + str(bbox[-9])+ ' ' + str(bbox[-8]) + ' ' + str(
                                        bbox[-7])+ ' ' + str(bbox[-6]) + ' ' +  str(bbox[-5])+ ' ' + str(bbox[-4]) + ' ' + str(bbox[-3])+ ' ' + str(
                                        bbox[-2]) + '\n')                         
                       

                       
                       
