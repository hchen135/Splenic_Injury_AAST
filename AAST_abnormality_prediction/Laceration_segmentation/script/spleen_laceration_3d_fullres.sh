export nnUNet_raw_data_base=/data/datasets/Spleen/nnUNet/nnUNet_raw_data_base;
export nnUNet_preprocessed=/data/datasets/Spleen/nnUNet/nnUNet_preprocessed;
export RESULTS_FOLDER=/data/datasets/Spleen/nnUNet/RESULTS_FOLDER;
nnUNet_train 3d_fullres nnUNetTrainerV2 Task501_Laceration 0 --npz;
nnUNet_train 3d_fullres nnUNetTrainerV2 Task501_Laceration 1 --npz;
nnUNet_train 3d_fullres nnUNetTrainerV2 Task501_Laceration 2 --npz;
nnUNet_train 3d_fullres nnUNetTrainerV2 Task501_Laceration 3 --npz;
nnUNet_train 3d_fullres nnUNetTrainerV2 Task501_Laceration 4 --npz;
