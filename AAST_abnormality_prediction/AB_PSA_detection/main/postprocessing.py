import json
import nibabel as nib
import os
import numpy as np
NUM_FOLD = 5
RESULT_DIR = "/data/datasets/Spleen/detectron2/ab_psa/output"
SPLN_AREA_DIR = "/data/datasets/Spleen/SpleenSegmentation/data/Dave_spleen"
#POST_FIX = ""
POST_FIX = "_mip_newdata"
box_dilation_size = 10

# we need to remove ab inside spleen and remove ab far away from spleen
def check(spln_area_data,box,slice_ind):
	slice_data = spln_area_data[:,:,slice_ind]
	box_data = slice_data[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
	n = np.sum(box_data)
	if n == 0:
		return 0
	elif n == np.prod(box_data.shape):
		return 1
	else:
		return 0.5


def post_processing(spln_area_ori_data,spln_area_postprocessed_data,file_name,boxes,scores):
	slice_ind = int(file_name.split('/')[-1].split('.')[0].split('_')[-1])
	new_boxes = []
	new_scores = []
	for ind,box in enumerate(boxes):
		if check(spln_area_ori_data,[box[0]-box_dilation_size,box[1]-box_dilation_size,box[2]+box_dilation_size,box[3]+box_dilation_size],slice_ind) == 1:
			continue
		if check(spln_area_postprocessed_data,box,slice_ind) == 0:
			continue
		new_boxes.append(box)
		new_scores.append(scores[ind])

	return new_boxes,new_scores





for fold in range(NUM_FOLD):
	with open(os.path.join(RESULT_DIR,"final_result_fold"+str(fold)+"_ab"+POST_FIX+".json")) as a:
		result_content = json.load(a)

	for subject in result_content:
		if 'spleen SOS' in subject:
			continue
		# load spln area segmentation
		try:
			spln_area_ori = nib.load(os.path.join(SPLN_AREA_DIR,subject,'PV','pv_spleen_area_ori_v6_'+subject.split(' ')[-1]+'_.nii'))
		except:
			spln_area_ori = nib.load(os.path.join(SPLN_AREA_DIR,subject,'pv','pv_spleen_area_ori_v6_'+subject.split(' ')[-1]+'_.nii'))
		try:
			spln_area_postprocessed = nib.load(os.path.join(SPLN_AREA_DIR,subject,'PV','pv_spleen_area_v6_'+subject.split(' ')[-1]+'_.nii'))
		except:
			spln_area_postprocessed = nib.load(os.path.join(SPLN_AREA_DIR,subject,'pv','pv_spleen_area_v6_'+subject.split(' ')[-1]+'_.nii'))

		spln_area_ori_data = spln_area_ori.get_fdata()
		spln_area_postprocessed_data = spln_area_postprocessed.get_fdata()

		# get all slice predictions
		slice_predictions = result_content[subject]["slice_predictions"]
		scores_final = 0

		for slice_prediction in slice_predictions:
			file_name = slice_prediction['file_name']
			boxes = slice_prediction['boxes']
			scores = slice_prediction['scores']

			new_boxes, new_scores = post_processing(spln_area_ori_data,spln_area_postprocessed_data,file_name,boxes,scores)
			slice_prediction['boxes'] = new_boxes
			slice_prediction['scores'] = new_scores

			if len(new_scores) == 0:
				slice_prediction['final_prediction'] = 0
			else:
				slice_prediction['final_prediction'] = np.max(new_scores)
			scores_final = max(scores_final,slice_prediction['final_prediction'])

		result_content[subject]["prediction"] = scores_final

	with open(os.path.join(RESULT_DIR,"final_result_fold"+str(fold)+"_postprocessed_ab"+POST_FIX+".json"),'w') as a:
		json.dump(result_content,a,indent=4)
