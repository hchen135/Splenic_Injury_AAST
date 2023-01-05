from glob import glob
import nrrd
import nibabel as nib
import numpy as np 
from skimage.measure import label, regionprops
from skimage.io import imsave
import skimage
import os
import json
from pycocotools import mask as maskUtils
from scipy.ndimage import binary_dilation
import pandas as pd
import warnings
warnings.filterwarnings('ignore') 

np.random.seed(1)

data_dir = "/ssd/Spleen/UNETR_spleen_lesion_seg_v2/BTCV/dataset/Dave_spleen_mip_npy"
SPLN_AREA_DIR = '/data/datasets/Spleen/SpleenSegmentation/data/Dave_spleen'
out_dir = '/data/datasets/Spleen/spleen niftis labeled KEY DATASET_COCO_ab_psa_mip_newdata'
external_dir = '/data/datasets/Spleen/spleen_external_data/splenic_vascular_injury_cases_cleaned'

clip_low = -80
clip_high = 360


subjects = glob(os.path.join(data_dir,'*/'))
csv = pd.read_csv('/data/datasets/Spleen/spleen niftis labeled KEY DATASET/UPDATED_spleen_project_dataset_deidentified_FINAL_KC_5.30.2021_-_updated_2.2.csv')
selected_ind = csv.values[csv.values[:,38] >=3,0]
assert len(selected_ind) == 83
#selected_ind = np.array([1,2,4,6,12,14]) # for debug
print(subjects[0])
external_subjects = glob(os.path.join(external_dir,'*/'))
external_range_file = os.path.join(external_dir,'spleen_area_hard_crop.json')
with open(external_range_file) as a:
	external_range_dict = json.load(a)

if not os.path.exists(out_dir+'/'+'images' ):
	os.makedirs(out_dir+'/'+'images')
if not os.path.exists(out_dir+'/'+'images'+'/'+'art' ):
	os.makedirs(out_dir+'/'+'images'+'/'+'art')
if not os.path.exists(out_dir+'/'+'images'+'/'+'pv' ):
	os.makedirs(out_dir+'/'+'images'+'/'+'pv')



train_subjects = np.random.choice(subjects,size = int(len(subjects)*0.7),replace=False).tolist()
test_subjects = [subject for subject in subjects if subject not in train_subjects]
assert (len(train_subjects) + len(test_subjects) == len(subjects))

art_images_train = []
art_annotations_train = []
art_images_test = []
art_annotations_test = []

pv_images_train = []
pv_annotations_train = []
pv_images_test = []
pv_annotations_test = []

art_categories = [
	{'name':'Pseudoaneurysm', 'id':1},
]

pv_categories = [
	{'name':'ActiveBleeding', 'id':1},
#	{'name':'Laceration', 'id':2},
#	{'name':'Hemoperitoneum', 'id':3},
]
art_image_count = 1
art_anno_count = 1
pv_image_count = 1
pv_anno_count = 1
for subject in subjects:
	print(subject)
	phase = 'train' if subject in train_subjects else 'test'
	if int(subject.split('/')[-2].split(' ')[-1]) not in selected_ind:
		continue
	try:
		art_img_path = glob(os.path.join(subject,'art','mip_img.npy'))[0]
		art = 1
	except:
		art_img_path = glob(os.path.join(subject,'ART','mip_img.npy'))[0]
		art = 2
	try:
		venus_img_path = glob(os.path.join(subject,'pv','mip_img.npy'))[0]
		pv = 1
	except:
		venus_img_path = glob(os.path.join(subject,'PV','mip_img.npy'))[0]
		pv = 2
	

	_name = 'art' if art == 1 else 'ART'
	art_anno_path = glob(os.path.join(subject,_name,'anno.npy'))[0]
	art_anno_data = np.load(art_anno_path)
	_name = 'pv' if pv == 1 else 'PV'
	venus_anno_path = glob(os.path.join(subject,_name,'anno.npy'))[0]
	venus_anno_data = np.load(venus_anno_path)

	art_img = np.load(art_img_path)
	venus_img = np.load(venus_img_path)

	art_shape = art_img.shape
	venus_shape = venus_img.shape

	_subject_name = art_img_path.split('/')[-3]
	_pv_name = venus_img_path.split('/')[-2]
	_art_name = art_img_path.split('/')[-2]
	
	art_spleen_area = nib.load(os.path.join(SPLN_AREA_DIR,_subject_name,_art_name,_art_name.lower()+'_spleen_area_v6_'+_subject_name.split(' ')[-1]+'_.nii')).get_fdata()
	pv_spleen_area = nib.load(os.path.join(SPLN_AREA_DIR,_subject_name,_pv_name,_pv_name.lower()+'_spleen_area_v6_'+_subject_name.split(' ')[-1]+'_.nii')).get_fdata()
	#art_spleen_area = binary_dilation(art_spleen_area,iterations=10)
	#pv_spleen_area = binary_dilation(pv_spleen_area,iterations=10)
	for z in range(art_shape[2]):
		if np.sum(art_spleen_area[:,:,z]) < 64:
			continue
		#first save original the images
		art_image_slice = ((np.clip(art_img[:,:,z],clip_low,clip_high)-clip_low)/(clip_high - clip_low)*255).astype(np.uint8)
		
		name_base = out_dir+'/'+'images' + '/'+'art'+'/'+art_img_path.split('/')[-3]+'_art_'+str(z)
		#dilated_spleen_area = binary_dilation(art_spleen_area[:,:,z],iterations=30)
		#art_image_slice[dilated_spleen_area == 0] = 0
		imsave(name_base+'.png',art_image_slice)

		#initialization preparation
		images_single = {}
		images_single['file_name'] = name_base+'.png'
		images_single['height'] = 512
		images_single['width'] = 512
		images_single['id'] = art_image_count


		globals()['art_images_'+phase].append(images_single)
		anno_final = art_anno_data[:,:,z] == 1
		# class_2: pseudoaneurysm on artery only
		#anno_final[art_anno_ori[0]:art_anno_ori[0]+art_anno_range[0],art_anno_ori[1]:art_anno_ori[1]+art_anno_range[1]] += art_anno_data[0,:,:,int(z-art_anno_ori[-1])]*1
		#anno_final[art_anno_ori[0]:art_anno_ori[0]+art_anno_range[0],art_anno_ori[1]:art_anno_ori[1]+art_anno_range[1]] += art_anno_data[1,:,:,int(z-art_anno_ori[-1])]*1
		#anno_final[art_anno_ori[0]:art_anno_ori[0]+art_anno_range[0],art_anno_ori[1]:art_anno_ori[1]+art_anno_range[1]] = np.clip( anno_final[art_anno_ori[0]:art_anno_ori[0]+art_anno_range[0],art_anno_ori[1]:art_anno_ori[1]+art_anno_range[1]],0,2)

		anno_final = anno_final.astype(int)
		label_image = label(anno_final)

		for region in regionprops(label_image, intensity_image = anno_final):
			# take regions with large enough areas
			if region.area > 0:
				anno_single = {}

				minr, minc, maxr, maxc = region.bbox
				rlength = maxr - minr
				clength = maxc - minc
				area = region.area
				category_id = int(region.max_intensity)
				assert category_id in [1.0], "category_id is {}".format(category_id)

				tmp_mask = np.zeros_like(label_image)
				tmp_mask[minr:maxr,minc:maxc] = region.image
				fortran_ground_truth_binary_mask = np.asfortranarray(tmp_mask.astype(np.uint8))
				encoded_ground_truth = maskUtils.encode(fortran_ground_truth_binary_mask)
				ground_truth_area = maskUtils.area(encoded_ground_truth)
				ground_truth_bounding_box = maskUtils.toBbox(encoded_ground_truth)
				contours = skimage.measure.find_contours(tmp_mask, 0.5)
				anno_single['segmentation'] = []
				for contour in contours:
					contour = np.flip(contour, axis=1)
					segmentation = contour.ravel().tolist()
					anno_single["segmentation"].append(segmentation)
				#r_index,c_index = np.where(region.image)
				#r_index = (r_index + minr).astype(int).tolist()
				#c_index = (c_index + minc).astype(int).tolist()

				#anno_single['segmentation'] = [r_index,c_index]
				anno_single['area'] = int(ground_truth_area)
				anno_single['iscrowd'] = 0
				anno_single['image_id'] = art_image_count
				anno_single['bbox'] = ground_truth_bounding_box.astype(int).tolist()
				anno_single['category_id'] = category_id
				anno_single['id'] = art_anno_count

				art_anno_count += 1

				globals()['art_annotations_'+phase].append(anno_single)	
		art_image_count += 1
		if (art_image_count%1000== 0):
			print(art_image_count)
	for z in range(venus_shape[2]):
		if np.sum(pv_spleen_area[:,:,z]) < 64:
			continue
		#first save original the images
		venus_image_slice = ((np.clip(venus_img[:,:,z],clip_low,clip_high)-clip_low)/(clip_high - clip_low)*255).astype(np.uint8)
		
		name_base = out_dir+'/'+'images' + '/'+'pv'+'/'+venus_img_path.split('/')[-3]+'_venus_'+str(z)
		#dilated_spleen_area = binary_dilation(pv_spleen_area[:,:,z],iterations=30)
		#venus_image_slice[dilated_spleen_area == 0] = 0
		imsave(name_base+'.png',venus_image_slice)

		#initialization preparation
		images_single = {}
		images_single['file_name'] = name_base+'.png'
		images_single['height'] = 512
		images_single['width'] = 512
		images_single['id'] = pv_image_count


		globals()['pv_images_'+phase].append(images_single)

		anno_final = venus_anno_data[:,:,z] == 1
		# class_1: active bleeding on venus only
		# class_3: laceration on venus only
		# class_6: hemoperitoneum on venus only no need for now
			#anno_final[venus_anno_ori[0]:venus_anno_ori[0]+venus_anno_range[0],venus_anno_ori[1]:venus_anno_ori[1]+venus_anno_range[1]] += venus_anno_data[0,:,:,int(z-venus_anno_ori[-1])]
			#if venus_anno_data.shape[0] >2:
			#	anno_final[venus_anno_ori[0]:venus_anno_ori[0]+venus_anno_range[0],venus_anno_ori[1]:venus_anno_ori[1]+venus_anno_range[1]] += venus_anno_data[2,:,:,int(z-venus_anno_ori[-1])]*2
			#	anno_final[venus_anno_ori[0]:venus_anno_ori[0]+venus_anno_range[0],venus_anno_ori[1]:venus_anno_ori[1]+venus_anno_range[1]] = np.clip(anno_final[venus_anno_ori[0]:venus_anno_ori[0]+venus_anno_range[0],venus_anno_ori[1]:venus_anno_ori[1]+venus_anno_range[1]],0,2)
			#if venus_anno_data.shape[0] >5:
			#	anno_final[venus_anno_ori[0]:venus_anno_ori[0]+venus_anno_range[0],venus_anno_ori[1]:venus_anno_ori[1]+venus_anno_range[1]] += venus_anno_data[5,:,:,int(z-venus_anno_ori[-1])]*3
			#	anno_final[venus_anno_ori[0]:venus_anno_ori[0]+venus_anno_range[0],venus_anno_ori[1]:venus_anno_ori[1]+venus_anno_range[1]] = np.clip(anno_final[venus_anno_ori[0]:venus_anno_ori[0]+venus_anno_range[0],venus_anno_ori[1]:venus_anno_ori[1]+venus_anno_range[1]],0,3)

		anno_final = anno_final.astype(int)
		label_image = label(anno_final)

		for region in regionprops(label_image, intensity_image = anno_final):
			# take regions with large enough areas
			if region.area >= 0:
				anno_single = {}

				minr, minc, maxr, maxc = region.bbox
				rlength = maxr - minr
				clength = maxc - minc
				area = region.area
				category_id = int(region.max_intensity)
				assert category_id in [1.0,2.0],"category_id is {}".format(category_id)

				tmp_mask = np.zeros_like(label_image)
				tmp_mask[minr:maxr,minc:maxc] = region.image
				fortran_ground_truth_binary_mask = np.asfortranarray(tmp_mask.astype(np.uint8))
				encoded_ground_truth = maskUtils.encode(fortran_ground_truth_binary_mask)
				ground_truth_area = maskUtils.area(encoded_ground_truth)
				ground_truth_bounding_box = maskUtils.toBbox(encoded_ground_truth)
				contours = skimage.measure.find_contours(tmp_mask, 0.5)
				anno_single['segmentation'] = []
				for contour in contours:
					contour = np.flip(contour, axis=1)
					segmentation = contour.ravel().tolist()
					anno_single["segmentation"].append(segmentation)
				#r_index,c_index = np.where(region.image)
				#r_index = (r_index + minr).astype(int).tolist()
				#c_index = (c_index + minc).astype(int).tolist()
				anno_single['area'] = int(ground_truth_area)
				anno_single['iscrowd'] = 0
				anno_single['image_id'] = pv_image_count
				anno_single['bbox'] = ground_truth_bounding_box.astype(int).tolist()
				anno_single['category_id'] = category_id
				anno_single['id'] = pv_anno_count

				pv_anno_count += 1

				globals()['pv_annotations_'+phase].append(anno_single)

		pv_image_count += 1
		if (pv_image_count%1000 == 0):
			print(pv_image_count)


for subject in external_subjects:
	print(subject)
	subject_name = subject.split('/')[-2]
	phase = 'train' if subject in train_subjects else 'test'
	art_img_path = os.path.join(subject,'art','scan.nii.gz')
	venus_img_path = os.path.join(subject,'PVP','scan.nii.gz')
	art_anno_path = os.path.join(subject,'art','anno.nii.gz')
	venus_anno_path = os.path.join(subject,'PVP','anno.nii.gz')
	
	art_data = nib.load(art_img_path)
	_art_flip = [art_data.affine[0,0]>0,art_data.affine[1,1] < 0,art_data.affine[2,2]<0]
	art_img = art_data.get_fdata()
	art_z_min = external_range_dict[subject_name]['art_range'][0]
	art_z_max = external_range_dict[subject_name]['art_range'][1]
	venus_data = nib.load(venus_img_path)
	_venus_flip = [venus_data.affine[0,0]>0,venus_data.affine[1,1] < 0,venus_data.affine[2,2]<0]
	venus_img = venus_data.get_fdata()
	venus_z_min = external_range_dict[subject_name]['PVP_range'][0]
	venus_z_max = external_range_dict[subject_name]['PVP_range'][1]
	
	art_anno_data = nib.load(art_anno_path)
	_art_anno_flip = [art_anno_data.affine[0,0]>0,art_anno_data.affine[1,1] < 0,art_anno_data.affine[2,2]<0]
	art_anno_img = art_anno_data.get_fdata()
	venus_anno_data = nib.load(venus_anno_path)
	_venus_anno_flip = [venus_anno_data.affine[0,0]>0,venus_anno_data.affine[1,1] < 0,venus_anno_data.affine[2,2]<0]
	venus_anno_img = venus_anno_data.get_fdata()

	# art
	for z in range(art_z_min,art_z_max+1):
		art_image_slice = ((np.clip(art_img[:,:,z],clip_low,clip_high)-clip_low)/(clip_high - clip_low)*255).astype(np.uint8)
		
		name_base = out_dir+'/'+'images' + '/'+'art'+'/'+subject_name+'_art_'+str(z)
		
		if _art_flip[0]:
			art_image_slice = art_image_slice[::-1,:]
		if _art_flip[1]:
			art_image_slice = art_image_slice[:,::-1]
		imsave(name_base+'.png',art_image_slice)
		
		#initialization preparation
		images_single = {}
		images_single['file_name'] = name_base+'.png'
		images_single['height'] = 512
		images_single['width'] = 512
		images_single['id'] = art_image_count

		globals()['art_images_'+phase].append(images_single)

		# anno
		anno_final = art_anno_img[:,:,z]
		if _art_anno_flip[0]:
			anno_final = anno_final[::-1,:]
		if _art_anno_flip[1]:
			anno_final = anno_final[:,::-1]
		anno_final = anno_final.astype(int)
		label_image = label(anno_final)
		
		for region in regionprops(label_image, intensity_image = anno_final):
			if region.area >= 0:
				anno_single = {}
				
				minr, minc, maxr, maxc = region.bbox
				rlength = maxr - minr
				clength = maxc - minc
				area = region.area
				category_id = 1
				tmp_mask = np.zeros_like(label_image)
				tmp_mask[minr:maxr,minc:maxc] = region.image
				fortran_ground_truth_binary_mask = np.asfortranarray(tmp_mask.astype(np.uint8))
				encoded_ground_truth = maskUtils.encode(fortran_ground_truth_binary_mask)
				ground_truth_area = maskUtils.area(encoded_ground_truth)
				ground_truth_bounding_box = maskUtils.toBbox(encoded_ground_truth)
				contours = skimage.measure.find_contours(tmp_mask, 0.5)
				anno_single['segmentation'] = []
				for contour in contours:
					contour = np.flip(contour, axis=1)
					segmentation = contour.ravel().tolist()
					anno_single["segmentation"].append(segmentation)
				
				anno_single['area'] = int(ground_truth_area)
				anno_single['iscrowd'] = 0
				anno_single['image_id'] = art_image_count
				anno_single['bbox'] = ground_truth_bounding_box.astype(int).tolist()
				anno_single['category_id'] = category_id
				anno_single['id'] = art_anno_count
		
				art_anno_count += 1

				globals()['art_annotations_'+phase].append(anno_single)
		art_image_count += 1
		if (art_image_count%1000== 0):
			print(art_image_count)

	for z in range(venus_z_min,venus_z_max+1):
		venus_image_slice = ((np.clip(venus_img[:,:,z],clip_low,clip_high)-clip_low)/(clip_high - clip_low)*255).astype(np.uint8)
		name_base = out_dir+'/'+'images' + '/'+'pv'+'/'+subject_name+'_venus_'+str(z)
		if _venus_flip[0]:
			venus_image_slice = venus_image_slice[::-1,:]
		if _venus_flip[1]:
			venus_image_slice = venus_image_slice[:,::-1]
		imsave(name_base+'.png',venus_image_slice)

		#initialization preparation
		images_single = {}
		images_single['file_name'] = name_base+'.png'
		images_single['height'] = 512
		images_single['width'] = 512
		images_single['id'] = pv_image_count
		
		globals()['pv_images_'+phase].append(images_single)
		
		# anno
		anno_final = venus_anno_img[:,:,z]
		if _venus_anno_flip[0]:
			anno_final = anno_final[::-1,:]
		if _venus_anno_flip[1]:
			anno_final = anno_final[:,::-1]
		
		anno_final = anno_final.astype(int)
		label_image = label(anno_final)

		for region in regionprops(label_image, intensity_image = anno_final):
			if region.area >= 0:
				anno_single = {}
				
				minr, minc, maxr, maxc = region.bbox
				rlength = maxr - minr
				clength = maxc - minc
				area = region.area
				category_id = 1
				tmp_mask = np.zeros_like(label_image)
				tmp_mask[minr:maxr,minc:maxc] = region.image
				fortran_ground_truth_binary_mask = np.asfortranarray(tmp_mask.astype(np.uint8))
				encoded_ground_truth = maskUtils.encode(fortran_ground_truth_binary_mask)
				ground_truth_area = maskUtils.area(encoded_ground_truth)
				ground_truth_bounding_box = maskUtils.toBbox(encoded_ground_truth)
				contours = skimage.measure.find_contours(tmp_mask, 0.5)
				anno_single['segmentation'] = []
				for contour in contours:
					contour = np.flip(contour, axis=1)
					segmentation = contour.ravel().tolist()
					anno_single["segmentation"].append(segmentation)
			
				anno_single['area'] = int(ground_truth_area)
				anno_single['iscrowd'] = 0
				anno_single['image_id'] = pv_image_count
				anno_single['bbox'] = ground_truth_bounding_box.astype(int).tolist()
				anno_single['category_id'] = category_id
				anno_single['id'] = pv_anno_count

				pv_anno_count += 1
				
				globals()['pv_annotations_'+phase].append(anno_single)
		pv_image_count += 1
		if (pv_image_count%1000== 0):
			print(pv_image_count)




art_content_train = {'images':art_images_train,'annotations':art_annotations_train,'categories':art_categories}
art_content_test = {'images':art_images_test,'annotations':art_annotations_test,'categories':art_categories}
pv_content_train = {'images':pv_images_train,'annotations':pv_annotations_train,'categories':pv_categories}
pv_content_test = {'images':pv_images_test,'annotations':pv_annotations_test,'categories':pv_categories}

art_content_all = {'images':art_images_train + art_images_test,'annotations':art_annotations_train + art_annotations_test,'categories':art_categories}
pv_content_all = {'images':pv_images_train + pv_images_test,'annotations':pv_annotations_train + pv_annotations_test,'categories':pv_categories}

# 5_fold_split
random_selected_ind = np.random.choice(selected_ind.tolist()+[i.split('/')[-2] for i in external_subjects],len(selected_ind)+len(external_subjects),replace=False)
print(random_selected_ind)
fold_ind_length = (len(selected_ind) + len(external_subjects))// 5
for i in range(4):
	globals()['fold_'+str(i)+'_ind'] = random_selected_ind[int(fold_ind_length*i):int(fold_ind_length*(i+1))]
globals()['fold_4_ind'] = random_selected_ind[int(fold_ind_length*4):]
print(fold_0_ind)
print(fold_1_ind)
print(fold_2_ind)
print(fold_3_ind)
print(fold_4_ind)
def assign_image_and_annotations(phase,fold_ind):
	image_ids_val_final = []
	images_val_final = []
	annotations_val_final = []
	images_train_final = []
	annotations_train_final = []	


	for image in globals()[phase+'_content_all']['images']:
		if 'spleen SOS' in image['file_name']:
			check_pattern = image['file_name'].split('/')[-1].split('_')[0]
		else:
			check_pattern = image['file_name'].split('/')[-1].split('_')[0].split(' ')[-1]
		if check_pattern in globals()['fold_'+str(fold_ind)+'_ind']:
			images_val_final.append(image)
			if image['id'] not in image_ids_val_final:
				image_ids_val_final.append(image['id'])
		else:
			images_train_final.append(image)
	print(len(images_val_final))
	print(len(images_train_final))
	for anno in globals()[phase+'_content_all']['annotations']:
		if anno['image_id'] in image_ids_val_final:
			annotations_val_final.append(anno)
		else:
			annotations_train_final.append(anno)
	val_dict = {'images':images_val_final,'annotations':annotations_val_final,'categories':globals()[phase+'_categories']}
	train_dict = {'images':images_train_final,'annotations':annotations_train_final,'categories':globals()[phase+'_categories']}
	return train_dict,val_dict

for i in range(5):
	globals()['psa_train_fold_'+str(i)],globals()['psa_val_fold_'+str(i)] = assign_image_and_annotations('art',i)
	globals()['ab_train_fold_'+str(i)],globals()['ab_val_fold_'+str(i)] = assign_image_and_annotations('pv',i)
	with open(os.path.join(out_dir,'coco_style_psa_train_fold_'+str(i)+'.json'),'w') as a:	
		json.dump(globals()['psa_train_fold_'+str(i)],a,indent=4)
	with open(os.path.join(out_dir,'coco_style_psa_val_fold_'+str(i)+'.json'),'w') as a:
		json.dump(globals()['psa_val_fold_'+str(i)],a,indent=4)
	with open(os.path.join(out_dir,'coco_style_ab_train_fold_'+str(i)+'.json'),'w') as a:
		json.dump(globals()['ab_train_fold_'+str(i)],a,indent=4)
	with open(os.path.join(out_dir,'coco_style_ab_val_fold_'+str(i)+'.json'),'w') as a:
		json.dump(globals()['ab_val_fold_'+str(i)],a,indent=4)





#with open(os.path.join(out_dir,'coco_style_psa_train.json'),'w') as a:
#	json.dump(art_content_train,a,indent=4)
#with open(os.path.join(out_dir,'coco_style_psa_test.json'),'w') as a:
#	json.dump(art_content_test,a,indent=4)
#
#with open(os.path.join(out_dir,'coco_style_ab_train.json'),'w') as a:
#	json.dump(pv_content_train,a,indent=4)
#with open(os.path.join(out_dir,'coco_style_ab_test.json'),'w') as a:
#	json.dump(pv_content_test,a,indent=4)




					

















