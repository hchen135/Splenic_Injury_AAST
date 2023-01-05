from glob import glob
import nibabel as nib
import json
import os

raw_data_dir = '/ssd/Spleen/UNETR_spleen_lesion_seg_v2/BTCV/dataset/Dave_spleen'
dest_dir = '/data/datasets/Spleen/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data/Task502_spleenpv'
subjects = [i for i in glob(os.path.join(raw_data_dir,'*','*')) if i.split('/')[-1].lower() == 'pv']
assert len(subjects) == 174

if not os.path.exists(dest_dir):
	os.mkdir(dest_dir)
if not os.path.exists(os.path.join(dest_dir,'imagesTr')):
	os.mkdir(os.path.join(dest_dir,'imagesTr'))
	os.mkdir(os.path.join(dest_dir,'labelsTr'))

json_dict = {
    "name":"spleenpv",
    "description": "Portal Venus spleen lesion Segmentation",
    "reference": "Dave",
    "tensorImageSize":"3D",
    "modality":{
        "0": "CT"
    },
    "labels":{
        "0":"Background",
        "1":"Active Bleeding",
        "2":"Laceration",
    },
    "numTraining":len(subjects),
    "numTest":0,
    "training":[],
    "test":[]
}




for subject in subjects:
	images = glob(os.path.join(subject,'*'))
	assert len(images) == 2
	CT_image = [i for i in images if 'anno' not in i.split('/')[-1]][0]
	label_image = [i for i in images if 'anno' in i.split('/')[-1]][0]

	patient_name = CT_image.split('/')[-3]
	# store the CT image
	CT_data = nib.load(CT_image)
	normal_CT_image_dest = os.path.join(dest_dir,'imagesTr',patient_name+'.nii.gz')
	CT_image_dest = os.path.join(dest_dir,'imagesTr',patient_name+'_0000.nii.gz')
	nib.save(CT_data,CT_image_dest)
	# store the label image
	label_data = nib.load(label_image)
	label_fdata = label_data.get_fdata()
	label_fdata[label_fdata == 3] = 0
	new_label_data = nib.Nifti1Image(label_fdata,label_data.affine,label_data.header)
	label_image_dest = os.path.join(dest_dir,'labelsTr',patient_name+'.nii.gz')
	nib.save(new_label_data,label_image_dest)
	#nib.save(label_data,label_image_dest)

	sample_dict = {"image":normal_CT_image_dest,"label":label_image_dest}
	json_dict["training"].append(sample_dict)

with open(os.path.join(dest_dir,"dataset.json"),"w") as a:
	json.dump(json_dict,a,indent=4)

