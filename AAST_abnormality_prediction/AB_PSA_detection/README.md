# AB PSA detection

## Introduction

We apply Faster RCNN to detect active bleeding and pseudoanyreusm. We use [detectron2](https://github.com/facebookresearch/detectron2). Please move ``addition/evaluator.py`` to ``detectron2/detectron2/evaluation/PatientAUCevaluator.py`` and change ``__init__.py`` to include it when installing detectron2 package. Please also download the pretrained ``faster_rcnn_X_101_32x8d_FPN_3x`` model into ``pretrained_model`` folder.

## Data Preparation
Run this in ``data_prep`` folder to generate the coco style annotation files for training:
```Shell
python CT2label.py
```
If using ``MIP`` (max intensity projection), run
```Shell
python CT2label_mip_newdata.py 
```

## Training
Training code is in ``main`` folder. Several yaml files are related to the model training, which is ``ab.yaml``, ``ab_mip_newdata.yaml`` and ``psa.yaml``.

Train the model with bash files, such as
```Shell
python run_ab.sh
```

## postprocessing
We remove the detections that are far away from the spleen area. In ``main`` folder, run
```Shell
python postprocessing.py
```

## Visualization
In visualization folder, we can also visualize the detection results by 
```Shell
python visualization.py
``