# Splenic Injury AAST
This is the code for the AAST grading fot Splenic Injury with whole body CT scans.

## Code structure
### Splenic area segmentation
Whole body CT scans are too large and contains clices that we are not care about. We used teacher student models to find the splenic area and only used the slices containing the splenic area for folowing processing.

The code is at [spleen_area_segmentation](https://github.com/hchen135/Splenic_Injury_AAST/tree/main/spleen_area_segmentation) folder.

### AAST grading
The AAST grading criterion reformulated and contains the analysis with the existence of active bleeds (AB) and pseudoaneurysms (PSA) and the volume of splenic parenchymal disruption (SPD). AB is detected in the portal venus phase; PSA is detected and SPD is segmented in arterial phase of CT scans. Finally, we use a rule-based algorithm to predict AAST grading by strictly following the AAST guideline instead of recalculating the best SPD volume threhsold.

We achieved AB and PSA detection with [detectron2](https://github.com/facebookresearch/detectron2). The code is at [AB_PSA_detection](https://github.com/hchen135/Splenic_Injury_AAST/tree/main/AAST_abnormality_prediction/AB_PSA_detection) folder.

We achieved SPD segmentation with [nnUNet](https://github.com/MIC-DKFZ/nnUNet). The code is at [Laceration_segmentation](https://github.com/hchen135/Splenic_Injury_AAST/tree/main/AAST_abnormality_prediction/Laceration_segmentation) folder.


We achieved rule-based AAST grading. The code is at [Rule_based_AAST_grade](https://github.com/hchen135/Splenic_Injury_AAST/tree/main/AAST_abnormality_prediction/Rule_based_AAST_grade).