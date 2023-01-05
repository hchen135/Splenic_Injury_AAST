import nibabel as nib
from glob import glob
import numpy as np
import os

image_path = glob('/data/datasets/Spleen/docker/SpleenSegmentation/data/Task09_Spleen/imagesTr/*')
label_path = glob('/data/datasets/Spleen/docker/SpleenSegmentation/data/Task09_Spleen/labelsTr/*')

out_image_path = '/data/datasets/Spleen/docker/SpleenSegmentation/data/Task09_Spleen/imageFlipped'
out_label_path = '/data/datasets/Spleen/docker/SpleenSegmentation/data/Task09_Spleen/labelFlipped'

def flip(input_path,output_path):

	for path in input_path:
		img = nib.load(path)
		name = path.split('/')[-1]
		header = img.header
		affine = img.affine
		data = img.get_fdata()
		flipped_data = data[::-1,:,:]
		affine[0,0] = -affine[0,0]
		flipped_image = nib.Nifti1Image(flipped_data,affine,header)
		nib.save(flipped_image,os.path.join(output_path,name))



flip(image_path,out_image_path)
flip(label_path,out_label_path)
