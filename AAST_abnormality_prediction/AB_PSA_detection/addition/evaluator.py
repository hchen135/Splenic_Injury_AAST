from detectron2.evaluation import DatasetEvaluator
import torch
import numpy as np
from pycocotools.coco import COCO
import logging
from detectron2.data import MetadataCatalog
from detectron2.utils.file_io import PathManager
import contextlib
import io
import os
import json
from sklearn.metrics import auc, roc_curve

class PatientAUCEvaluator(DatasetEvaluator):
    def __init__(
        self,
        dataset_name,
        distributed=True,
        output_dir=None,
        cfg=None,
    ):
        self._logger = logging.getLogger(__name__)
        #self._logger.setLevel(logging.DEBUG)
        self._distributed = distributed
        self._dataset_name = dataset_name
        self._output_dir = output_dir
        self.cfg = cfg
        self.fold = int(cfg.DATASETS.TRAIN[0][-1])

        self._cpu_device = torch.device("cpu")
        
        #self.input_file_to_gt_file = {
        #    dataset_record["file_name"]: dataset_record["sem_seg_file_name"]
        #    for dataset_record in DatasetCatalog.get(dataset_name)
        #}
   
        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            if output_dir is None:
                raise ValueError(
                    "output_dir must be provided to COCOEvaluator "
                    "for datasets not in COCO format."
                )
            self._logger.info(f"Trying to convert '{dataset_name}' to COCO format ...")

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            convert_to_coco_json(dataset_name, cache_path, allow_cached=allow_cached_coco)

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file) # self._coco_api is the ground truth!
            self.cat_ids = self._coco_api.getCatIds()
            self.category_names = [self._coco_api.loadCats(i)[0]["name"] for i in range(1,1+cfg.MODEL.ROI_HEADS.NUM_CLASSES)]

        
    def reset(self):
        self._predictions = {}

    def store(self,final_dict):
        out_path = os.path.join('output','final_result_fold'+str(self.fold)+'.json')
        with open(out_path,'w') as a:
            json.dump(final_dict,a,indent=4)

    def process(self,inputs,outputs):
        cfg = self.cfg
        for input, output in zip(inputs,outputs):
            #### pred_operation
            prediction = {"image_id": input["image_id"],"file_name":input["file_name"]}
            instances = output["instances"].to(self._cpu_device)
            #prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if len(instances.scores) == 0:
                pred_score = 0.0
            else:
                pred_score = np.max(instances.scores.numpy())
        
            #### GT_operation
            anns_ids = self._coco_api.getAnnIds(imgIds=input['image_id'], catIds=self.cat_ids, iscrowd=None)
            anns = self._coco_api.loadAnns(anns_ids)
            GT_class = 1 if len(anns) > 0 else 0

            #### store result
            prediction["GT_class"] = int(GT_class)
            prediction["final_prediction"] = np.float64(pred_score)
            prediction["boxes"] = instances.pred_boxes.tensor.numpy().astype(np.float64).tolist()
            prediction["scores"] = instances.scores.numpy().astype(np.float64).tolist()
            prediction["file_name"] = input["file_name"]
 
            subject = input["file_name"].split('/')[-1].split('_')[0]
            if subject not in self._predictions.keys():
                self._predictions[subject] = []
            self._predictions[subject].append(prediction)
             


    def evaluate(self):
        cfg = self.cfg
        print(self._logger)
        self._logger.setLevel(logging.INFO)
        print(self._logger)
        print(__name__)
        print(self._logger)
        print(self._logger)
        print(self._logger)
        print(self._logger)
        print(self._logger)
        
        #### First convert slice GT classes and predictions into patient level
        GT_all = []
        prediction_all = []
        final_dict = {}
        for subject in self._predictions.keys():
            patient_GT_class = 0
            patient_prediction = 0
            boxes = []
            scores = []
            subject_single_dict = {"slice_predictions":[]}
            for instance in self._predictions[subject]:
                patient_GT_class = patient_GT_class or instance["GT_class"]
                patient_prediction = max(patient_prediction,instance["final_prediction"])
                subject_single_dict["slice_predictions"].append(instance)
            GT_all.append(int(patient_GT_class))
            prediction_all.append(patient_prediction)
            self._logger.warning(
			f"GT_class: {patient_GT_class}, "
			f"prediction: {patient_prediction}, "
			f"subject: {subject}")
            self._logger.log(logging.INFO,"GT_class: "+str(patient_GT_class)+", prediction: "+str(patient_prediction)+", subject: "+subject)
            subject_single_dict["GT_class"] = int(patient_GT_class)
            subject_single_dict["prediction"] = np.float64(patient_prediction)
            final_dict[subject] =  subject_single_dict
        
        fpr, tpr, _ = roc_curve(GT_all,prediction_all)
        roc_auc = auc(fpr,tpr)
        self._logger.log(logging.INFO,"AUCe score for test scans: "+str(roc_auc))
        
        self.store(final_dict)
        
  
                

        
