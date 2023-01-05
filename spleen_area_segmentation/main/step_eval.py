from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
import nibabel as nib
import argparse
import time

parser = argparse.ArgumentParser(description='delete naiive false positive segementations predicted')
parser.add_argument('--version', default=1,type=int, help='which iteration to process')
parser.add_argument('--threshold', default=0,type=float, help='logit threshold for positive class extraction')
args = parser.parse_args()
version = args.version
assert version >1
print('Version: ',version)
print('Threshold: ',args.threshold)
ROOT = "/data/datasets/Spleen/SpleenSegmentation/data/Dave_spleen"
OUTPUT_DIR  ='/data/datasets/Spleen/SpleenSegmentation/data/Dave_spleen'
#for image_path in [i for i in glob.glob(os.path.join(ROOT,'*','*')) if len(i.split('/')[-2].split(' ')) == 3]:
#    paths = glob.glob(os.path.join(image_path,'*.nii'))
#    if len(paths) != 1:
#        print(image_path)
images_path = [i for i in glob.glob(os.path.join(ROOT,'*','*','*.nii')) if len(i.split('/')[-3].split(' ')) == 3 and 'spleen_area' not in i]
assert len(images_path) == 174 * 2
#images_path = ["/data/datasets/Spleen/spleen niftis labeled KEY DATASET/spln inj 1/art/Splenic_Injury1_1_1_CT_HEADBRAIN_WO_CON_DE_BODY_4.0_Br40_3_F_0.7_13.nii"]
data_dicts = [{'image':image_path,'name':image_path} for image_path in images_path]
print('start loading dataset')
# transforms same as training
val_org_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        #Spacingd(keys=["image"], pixdim=(
        #    1.5, 1.5, 2.0), mode="bilinear"),
        #Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-80, a_max=320,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        #CropForegroundd(keys=["image"], source_key="image"),
        EnsureTyped(keys=["image"]),
    ]
)

# create data loader
val_ds = Dataset(data=data_dicts, transform=val_org_transforms)
print(len(val_ds))
val_loader = DataLoader(val_ds, batch_size=1, num_workers=8)

# create model
device = torch.device("cuda:0")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)

start_time = time.time()
pretrained_model_name = os.path.join("/data/datasets/Spleen/SpleenSegmentation","weights", "step"+str(version)+"_best_metric_model.pth")
model.load_state_dict(torch.load(pretrained_model_name))
model.eval()
print('start evaluation')
with torch.no_grad():
    for i, val_data in enumerate(val_loader):
        batch_time_start = time.time()
        roi_size = (512, 512, 16)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(
            val_data["image"].to(device), roi_size, sw_batch_size, model
        )
        assert val_outputs.shape[0] == 1
        assert len(val_data['name']) == 1
        true_volume = nib.load(val_data['name'][0])

        print(true_volume.shape)
        print(val_outputs.shape)
        print(val_data['image'][0].shape)
        assert val_outputs.shape[2:] == true_volume.shape,'volume of outputs {} is not the same as the volume of the image'.format(val_outputs.shape[2:])
        #val_outputs = torch.argmax(val_outputs,dim=1).detach().cpu().numpy()[0]
        #val_outputs = torch.softmax(val_outputs,dim=1)
        val_outputs = (val_outputs[:,1] - val_outputs[:,0] >= args.threshold).detach().cpu().numpy()[0]
        val_outputs = nib.Nifti1Image(val_outputs, true_volume.affine, true_volume.header)
        subject = val_data['name'][0].split('/')[-3]
        phase = val_data['name'][0].split('/')[-2]
        if not os.path.exists(os.path.join(OUTPUT_DIR,subject,phase)):
            os.makedirs(os.path.join(OUTPUT_DIR,subject,phase))
        with open("log_step.txt","a") as a:
            a.write("batch time for "+str(val_data["image"].shape[0])+" is: "+str(time.time()-batch_time_start)+"\n")
        nib.save(val_outputs,os.path.join(OUTPUT_DIR,subject,phase,phase.lower()+'_spleen_area_ori_v'+str(version)+'_'+str(subject.split(' ')[-1])+'_.nii'))
print('evaluation finished')        
print("time used:",time.time() - start_time)
