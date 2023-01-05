from pycocotools.coco import COCO
import os
import cv2
from skimage.io import imread,imsave
import numpy as np
import json

from util import find_ori_image_info,find_annotations,draw_boxes,draw_masks

GT_DIR = '/data/datasets/Spleen/spleen niftis labeled KEY DATASET_COCO_ab_psa'
PRED_DIR = '/data/datasets/Spleen/detectron2/ab_psa/output'
OUT_DIR = './images'

NUM_FOLD = 5

phase_list = ['ab','psa']

for phase in phase_list:
	if not os.path.exists(os.path.join(OUT_DIR,phase)):
		os.mkdir(os.path.join(OUT_DIR,phase))
	for fold in range(NUM_FOLD):
		GT_json = os.path.join(GT_DIR,'coco_style_'+phase+'_val_fold_'+str(fold)+'.json')
		pred_json = os.path.join(PRED_DIR,'final_result_fold'+str(fold)+'_'+phase+'.json')
		with open(GT_json) as a:
			GT_content = json.load(a)
		with open(pred_json) as a:
			pred_content = json.load(a)
		coco = COCO(GT_json)

		for subject in pred_content:
			if 'spleen SOS' in subject:
				continue
			slice_predictions_list = pred_content[subject]
			for image_prediction in slice_predictions_list['slice_predictions']:
				image_id = image_prediction['image_id']
				ori_image_info = find_ori_image_info(GT_content,image_id)
				annotations = find_annotations(GT_content,image_id)				
			
				if len(image_prediction['boxes']) or len(annotations):
					ori_image = imread(image_prediction['file_name'])
					if len(ori_image.shape) == 2:
						ori_image = np.stack([ori_image]*3,axis=2)
					pred_image = draw_boxes(image_prediction,phase)
					if len(annotations):
						print(image_prediction['file_name'])
						GT_image = draw_masks(coco,image_id)
					else:
						GT_image = ori_image

					image_all = np.concatenate([ori_image,GT_image,pred_image],axis=1)
					imsave(os.path.join(OUT_DIR,phase,image_prediction['file_name'].split('/')[-1]),image_all)

					
		

