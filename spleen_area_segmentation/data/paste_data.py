from glob import glob
import os
import shutil
ori_image_dir = "/data/datasets/Spleen/spleen niftis labeled KEY DATASET"
dest_dir = "/data/datasets/Spleen/docker/SpleenSegmentation/data/Dave_spleen"
image_path = [i for i in glob(os.path.join(ori_image_dir,'*','*','*.nii')) if len(i.split('/')[-3].split(' ')) == 3]
print(len(image_path))

for path in image_path:
    dest_path = os.path.join(dest_dir,'/'.join(path.split('/')[-3:]))
    if not os.path.exists('/'.join(dest_path.split('/')[:-1])):
        os.makedirs('/'.join(dest_path.split('/')[:-1]))

    shutil.copyfile(path,dest_path)

