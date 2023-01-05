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
    RandScaleIntensityd,
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
import numpy as np 
import time
from util import choose_ind
import argparse
def printlog(string,filename="log_step.txt"):
    print(string)
    with open(filename,'a') as a:
        a.write(string+'\n')


parser = argparse.ArgumentParser(description='delete naiive false positive segementations predicted')
parser.add_argument('--version', default=1,type=int, help='which iteration to process')
args = parser.parse_args()
version = args.version
assert version >1
printlog('Version: '+str(version))



np.random.seed(1)


root_dir = "/data/datasets/Spleen/SpleenSegmentation/"
ROOT_TEACHER = "/data/datasets/Spleen/SpleenSegmentation/data/Task09_Spleen"
ROOT_STUDENT = "/data/datasets/Spleen/SpleenSegmentation/data/Dave_spleen"

train_images_student = sorted([i for i in glob.glob(os.path.join(ROOT_STUDENT,'*','*','*.nii')) if i.split('/')[-1][-5:] != '_.nii'])
train_images_teacher = sorted(glob.glob(os.path.join(ROOT_TEACHER, "imageFlipped", "*.nii.gz")))
if version == 2:
    train_labels_student = sorted([i for i in glob.glob(os.path.join(ROOT_STUDENT,'*','*','*.nii')) if i.split('/')[-1][-5:] == '_.nii' and '_v' not in i.split('/')[-1]])
else:
    train_labels_student = sorted([i for i in glob.glob(os.path.join(ROOT_STUDENT,'*','*','*.nii')) if i.split('/')[-1][-5:] == '_.nii' and '_v'+str(version-1) in i.split('/')[-1]])
train_labels_teacher = sorted(glob.glob(os.path.join(ROOT_TEACHER, "labelFlipped", "*.nii.gz")))
start_ind,end_ind = choose_ind(version)
print((start_ind,end_ind))
student_select_ind = np.random.choice(len(train_images_student),len(train_images_student),replace=False)[int(start_ind):int(end_ind)]
train_images = train_images_teacher + [train_images_student[i] for i in range(len(train_images_student)) if i in student_select_ind]
train_labels = train_labels_teacher + [train_labels_student[i] for i in range(len(train_labels_student)) if i in student_select_ind]
#train_images = train_images_teacher + train_images_student
#train_labels = train_labels_teacher + train_labels_student
data_dicts = [{"image": image_name, "label": label_name, "name":image_name}
    for image_name, label_name in zip(train_images, train_labels)]
assert len(train_images) == len(train_labels)
print(len(train_images))
val_inds = np.random.choice(len(train_images),12,replace=False)
train_files = [data_dicts[i] for i in range(len(data_dicts)) if i not in val_inds]
val_files = [data_dicts[i] for i in range(len(data_dicts)) if i in val_inds]
print(len(train_files))

set_determinism(seed=0)

train_transforms = Compose(
    [
        LoadImaged(keys=["image"],dtype=np.int16),
        LoadImaged(keys=["label"],dtype=np.int8),
        EnsureChannelFirstd(keys=["image", "label"]),
        #    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        #Orientationd(keys=["image", "label"], axcodes="LAS"),
        #ScaleIntensityRanged(
        #    keys=["image"], a_min=-80, a_max=320,
        #    b_min=0.0, b_max=1.0, clip=True,
        #),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(512, 512, 16),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        ScaleIntensityRanged(
            keys=["image"], a_min=-80, a_max=320,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        # user can also add other random transforms
        # RandAffined(
        #     keys=['image', 'label'],
        #     mode=('bilinear', 'nearest'),
        #     prob=1.0, spatial_size=(96, 96, 96),
        #     rotate_range=(0, 0, np.pi/15),
        #     scale_range=(0.1, 0.1, 0.1)),
        RandScaleIntensityd(keys="image",
                                       factors=0.05,
                                       prob=0.1),
        EnsureTyped(keys=["image", "label"],dtype=torch.float32),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image"],dtype=np.int16),
        LoadImaged(keys=["label"],dtype=np.int8),
        EnsureChannelFirstd(keys=["image", "label"]),
        #Spacingd(keys=["image", "label"], pixdim=(
        #    1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        #Orientationd(keys=["image", "label"], axcodes="LAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-80, a_max=320,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"],dtype=torch.float32),
    ]
)



# check_ds = Dataset(data=val_files, transform=val_transforms)
# check_loader = DataLoader(check_ds, batch_size=1)
# check_data = first(check_loader)
# image, label = (check_data["image"][0][0], check_data["label"][0][0])
# print(f"image shape: {image.shape}, label shape: {label.shape}")
# print('image_name',check_data["name"])
# # plot the slice [:, :, 80]
# plt.figure("check", (12, 6))
# plt.subplot(1, 2, 1)
# plt.title("image")
# plt.imshow(image[:, :, 27], cmap="gray")
# plt.subplot(1, 2, 2)
# plt.title("label")
# plt.imshow(label[:, :, 27])

# plt.savefig('check.png')

train_ds = CacheDataset(
    data=train_files, transform=train_transforms,
    cache_num=121, num_workers=8)
# train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=8)

val_ds = CacheDataset(
    data=val_files, transform=val_transforms,
    cache_rate=1.0, num_workers=8)
# val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=8)

device = torch.device("cuda:0")
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
    #dropout=0.1
).to(device)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

time_start = time.time()
# training
max_epochs = 600
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

time_start = time.time()
for epoch in range(max_epochs):
    printlog("-" * 10)
    printlog("epoch {}/{}".format(epoch+1,max_epochs))
    time_epochstart = time.time()
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        printlog( "{}/{}, ".format(step,len(train_ds) // train_loader.batch_size) + "train_loss: {:.4f}".format(loss.item()))
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    printlog("epoch {} average loss: {:.4f}, time: {:.4f}".format(epoch+1,epoch_loss,time.time()-time_epochstart))

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (512, 512, 16)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, model)
                val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                # compute metric for current iteration
                dice_metric(y_pred=val_outputs, y=val_labels)

            # aggregate the final mean dice result
            metric = dice_metric.aggregate().item()
            # reset the status for next validation round
            dice_metric.reset()

            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    root_dir,"weights", "step"+str(version)+"_best_metric_model.pth"))
                printlog("saved new best metric model")
            printlog(
                "current epoch: {} current mean dice: {:.4f}".format(epoch + 1,metric)+"\nbest mean dice: {:.4f} ".format(best_metric)+"at epoch: {}".format(best_metric_epoch)
            )


printlog(
    "train completed, best_metric: {:.4f} ".format(best_metric)+"at epoch: {}".format(best_metric_epoch)+", total time: {:.4f}".format(time.time()-time_start))

#inspection
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)

plt.savefig('metrix.png')

# model.load_state_dict(torch.load(
#     os.path.join(root_dir, "weights","best_metric_model.pth")))
# model.eval()
# with torch.no_grad():
#     for i, val_data in enumerate(val_loader):
#         roi_size = (512, 512, 16)
#         sw_batch_size = 4
#         val_outputs = sliding_window_inference(
#             val_data["image"].to(device), roi_size, sw_batch_size, model
#         )
#         # plot the slice [:, :, 80]
#         plt.figure("check", (18, 6))
#         plt.subplot(1, 3, 1)
#         plt.title(f"image {i}")
#         plt.imshow(val_data["image"][0, 0, :, :, 27], cmap="gray")
#         plt.subplot(1, 3, 2)
#         plt.title(f"label {i}")
#         plt.imshow(val_data["label"][0, 0, :, :, 27])
#         plt.subplot(1, 3, 3)
#         plt.title(f"output {i}")
#         plt.imshow(torch.argmax(
#             val_outputs, dim=1).detach().cpu()[0, :, :, 27])
#         plt.savefig('validation.png')
#         if i == 2:
#             break

print('time spent: ', time.time()-time_start)

