# Laceration Segmentation

## Introduction

We apply [nnUNet](https://github.com/MIC-DKFZ/nnUNet) to segment laceration

## Data preparation

Please refer to [nnUNet](https://github.com/MIC-DKFZ/nnUNet) or ``data_prep/data_convertion_Task502_spleenpv.py`` to prepare the data.

## Training
Run 2D nnUNet model
```Shell
sh spleen_laceration_2d.sh
```

Run 3D_fullres nnUNet model
```Shell
sh spleen_laceration_3d_fullres.sh
```

Run 3D_cascade nnUNet model
```Shell
sh spleen_laceration_3d_cascade.sh
```

Run 3D_cascade_fullres nnUNet model
```Shell
sh spleen_laceration_3d_cascade_fullres.sh
```

Finally, choose the bese ensemble results with
```Shell
sh spleen_laceration_model_choosing.sh
```

The results are saved in the ``RESULTS_FOLDER``.