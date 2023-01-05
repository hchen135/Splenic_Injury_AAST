export nnUNet_raw_data_base=/data/datasets/Spleen/nnUNet/nnUNet_raw_data_base;
export nnUNet_preprocessed=/data/datasets/Spleen/nnUNet/nnUNet_preprocessed;
export RESULTS_FOLDER=/data/datasets/Spleen/nnUNet/RESULTS_FOLDER;

nnUNet_find_best_configuration -m 2d 3d_fullres 3d_lowres 3d_cascade_fullres -t 501
