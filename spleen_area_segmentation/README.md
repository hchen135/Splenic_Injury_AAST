# Spleen Area Segmentation

## Introduction

We apply teacher-student model to segment the splenic areas in the internal dataset. We use the external dataset wit spleen annotations as the teacher. The method is from [NoisyStudent](https://github.com/sally20921/NoisyStudent).

### Data Preparation

We first copy the raw data into a folder of this project by ``data/paste_data.py``.

### Training

We first need to train the model with external dataset alone by
```Shell
python train.py
```

Then we get the spleen area segmentation results with 
```Shell
python step_eval.py --version=1
```

Next step is to train the teacher-student model. We use the students' predictions as the annotations and train together with the teacher dataset. Then we predict the students' spleen area again. Due to the slow CPU I/O speed, we only train a portion of student data every time. So we need to train the model for several steps to see all the student data. Every time, we same the student prediction into different ``version``.

For each version ``$V$``, we first train with 
```Shell
python step.py --version=$V$
```
Then predict with 
```Shell
python step_eval.py --version=$V$
```
We applied 6 times. 

Please be aware that ``version=2`` may have some mismatch problem in ``step.py``. Users could delete line 63 in ``step.py`` and delete the if/else code.
