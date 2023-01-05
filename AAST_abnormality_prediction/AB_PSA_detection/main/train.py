from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
)

from detectron2.data import DatasetCatalog

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
os.environ["NCCL_DEBUG"] = "INFO"
import time
import torch
import copy
import skimage
from pycocotools.coco import COCO
import numpy as np
from util import *

from detectron2 import model_zoo
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, DefaultPredictor, default_argument_parser, default_setup, launch
from detectron2.evaluation import CityscapesSemSegEvaluator, DatasetEvaluators, SemSegEvaluator, COCOEvaluator, PatientAUCEvaluator
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler

#from evaluator import PatientAUCEvaluator

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # add_deeplab_config(cfg)
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    #add_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)
    # cfg.merge_from_file(args.config_file)
    #cfg.MODEL.ANCHOR_GENERATOR.SIZES= [[8], [16],[32], [64], [128]]
    #cfg.DATASETS.TRAIN = ("spleen_pv_train",)
    #cfg.DATASETS.TEST = ("spleen_pv_test",)
    #cfg.INPUT.MIN_SIZE_TRAIN = 512
    # cfg.merge_from_list(args.opts)
    # cfg.DATALOADER.NUM_WORKERS = 2
    #cfg.MODEL.WEIGHTS = './pretrained_model/model_final_f10217.pkl'  # the original one is the deeplab sample script
    #cfg.SOLVER.IMS_PER_BATCH = 32
    #cfg.SOLVER.BASE_LR = 0.005  # pick a good LR
    #cfg.SOLVER.MAX_ITER = 60000     # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    #cfg.SOLVER.STEPS = (30000,)
    # cfg.SOLVER.STEPS = []        # do not decay learning rate
    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    #cfg.SPLASH = True
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    print(cfg.DATASETS.TRAIN)
    fold = int(cfg.DATASETS.TRAIN[0].split('fold')[-1].split('_')[0])
    task = cfg.DATASETS.TRAIN[0].split('_')[1]
    mip = True if 'mip' in cfg.DATASETS.TRAIN[0] else False
    phase = 'pv' if task =='ab' else 'art'
    if not mip:
        register_coco_instances("spleen_"+task+"_train_fold"+str(fold), {}, "/data/datasets/Spleen/spleen niftis labeled KEY DATASET_COCO_ab_psa/coco_style_"+task+"_train_fold_"+str(fold)+".json", "/data/datasets/Spleen/spleen niftis labeled KEY DATASET_COCO_ab_psa/images/"+phase+"/")
        register_coco_instances("spleen_"+task+"_test_fold"+str(fold), {}, "/data/datasets/Spleen/spleen niftis labeled KEY DATASET_COCO_ab_psa/coco_style_"+task+"_val_fold_"+str(fold)+".json", "/data/datasets/Spleen/spleen niftis labeled KEY DATASET_COCO_ab_psa/images/"+phase+"/")
    else:
        register_coco_instances("spleen_"+task+"_train_fold"+str(fold)+"_mip_newdata", {}, "/data/datasets/Spleen/spleen niftis labeled KEY DATASET_COCO_ab_psa_mip_newdata/coco_style_"+task+"_train_fold_"+str(fold)+".json", "/data/datasets/Spleen/spleen niftis labeled KEY DATASET_COCO_ab_psa_mip_newdata/images/"+phase+"/")
        register_coco_instances("spleen_"+task+"_test_fold"+str(fold)+"_mip_newdata", {}, "/data/datasets/Spleen/spleen niftis labeled KEY DATASET_COCO_ab_psa_mip_newdata/coco_style_"+task+"_val_fold_"+str(fold)+".json", "/data/datasets/Spleen/spleen niftis labeled KEY DATASET_COCO_ab_psa_mip_newdata/images/"+phase+"/")
     
    if args.eval_only:
        model = DefaultTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res
    
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    res = trainer.train()

    #res = trainer.test(cfg,trainer.model,COCOEvaluator(cfg.DATASETS.TEST[0],output_dir=cfg.OUTPUT_DIR))
    res = trainer.test(cfg,trainer.model,PatientAUCEvaluator(cfg.DATASETS.TEST[0],output_dir=cfg.OUTPUT_DIR,cfg=cfg))
    return res



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        args=(args,),
        dist_url=args.dist_url,
    )
