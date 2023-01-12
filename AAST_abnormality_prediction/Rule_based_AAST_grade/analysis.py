import pandas as pd 
import numpy as np 
from sklearn.linear_model import LogisticRegression 

from util import acc_calc,cohen_kappa,AE_auc,pearson_score,relative_volume_difference,volum_similarity

# AB reuslts
# Sensitivity Specificity threshold 
# 0.8077      0.8378      0.178
# 0.7692      0.8581      0.268

# PSA results
# Sensitivity Specificity threshold 
# 0.8913      0.8594      0.108
# 0.8043      0.8906      0.412
# 0.7609      0.9063      0.46

AB_thresh = 0.178
PSA_thresh = 0.108



result_df = pd.read_csv("../processed_data.csv")
result_data = result_df.values
columns = result_df.columns.values
column_dict = {}
for ind,column in enumerate(columns):
	column_dict[column] = int(ind)

# Initialization
final_decisions = {}
for ind,patient in enumerate(result_data[:,column_dict['patient_id']]):
	final_decisions[patient] = {}
	final_decisions[patient]["pred"] = -1
	final_decisions[patient]["GT"] = result_data[ind,column_dict['grade_GT']]
	final_decisions[patient]["GT_ab"] = result_data[ind,column_dict['AB_GT']]
	final_decisions[patient]["GT_psa"] = result_data[ind,column_dict['PSA_GT']]
	final_decisions[patient]["pred_lac_vol"] = result_data[ind,column_dict['laceration_volume']]
	if result_data[ind,column_dict['grade_GT']] < 3:
		final_decisions[patient]["GT"] = 2

# First to get active bleeding and pseudoaneurysm gradings.
for ind,patient in enumerate(result_data[:,column_dict['patient_id']]):
	if result_data[ind,column_dict['PSA_prediction']] > PSA_thresh:
		final_decisions[patient]["pred"] = 4
	if result_data[ind,column_dict['AB_prediction']] > AB_thresh:
		final_decisions[patient]["pred"] = 5
acc,existence_acc = acc_calc(final_decisions,[4,5])
print("After diagnosing AB and PSA, accuracy of grade 4+5 is:",acc,", existence acc metric:",existence_acc)

# Then we need to classifiy according to splenic area volume.
remaining_patients = [patient for patient in final_decisions if final_decisions[patient]["pred"] == -1]
X = []
y = []
for ind,patient in enumerate(result_data[:,column_dict['patient_id']]):
	if patient in remaining_patients:
		X.append([result_data[ind,column_dict['spleen_area_volume']],result_data[ind,column_dict['spleen_area_volume']]/result_data[ind,column_dict['CT_volume']]])
		y.append(result_data[ind,column_dict['grade_GT']] >= 4)
		if result_data[ind,column_dict['grade_GT']] >= 4:
			print(patient,result_data[ind,column_dict['grade_GT']])

clf = LogisticRegression(random_state=0).fit(X, y)
Y = clf.predict(X)
for ind in range(len(Y)):
	if Y[ind] == 1:
		final_decisions[remaining_patients[ind]]["pred"] = 4

acc,existence_acc = acc_calc(final_decisions,[4,5])
print("After diagnosing splenic area, accuracy of grade 4+5 is:",acc,", existence acc metric:",existence_acc)
# acc,existence_acc = acc_calc(final_decisions,[5])
# print(acc,existence_acc)

# Finally we need to classify according to laceration volume.
remaining_patients = [patient for patient in final_decisions if final_decisions[patient]["pred"] == -1]
X = []
y = []
for ind,patient in enumerate(result_data[:,column_dict['patient_id']]):
	if patient in remaining_patients:
		X.append([result_data[ind,column_dict['laceration_volume']]])
		if result_data[ind,column_dict['grade_GT']] < 3:
			y.append(0)
		elif result_data[ind,column_dict['grade_GT']] == 3:
			y.append(1)
		else:
			y.append(2)

clf = LogisticRegression(random_state=0).fit(X, y)
Y = clf.predict(X)
print(np.unique(Y))
for ind in range(len(Y)):
	if Y[ind] == 0:
		final_decisions[remaining_patients[ind]]["pred"] = 2
	elif Y[ind] == 1:
		final_decisions[remaining_patients[ind]]["pred"] = 3
	else:
		final_decisions[remaining_patients[ind]]["pred"] = 4

acc,existence_acc = acc_calc(final_decisions,[4,5])
print("After diagnosing laceration, accuracy of grade 4+5 is:",acc,", existence acc metric:",existence_acc)

acc,existence_acc = acc_calc(final_decisions,[3,4,5])
print("After diagnosing laceration, accuracy of grade 3+4+5 is:",acc,", existence acc metric:",existence_acc)

acc,existence_acc = acc_calc(final_decisions,[2,3,4,5])
print("After diagnosing laceration, accuracy of grade 2+3+4+5 is:",acc,", existence acc metric:",existence_acc)

print(np.unique(Y))
print(np.unique(y))
print(clf.coef_.T)
print(clf.intercept_)
'''
new_X = []
new_X = np.array([i*(15000-9928)/1000+9928 for i in range(1000)]).reshape(-1,1)
print(new_X.shape)
new_Y = clf.predict(new_X)
for i in range(1000):
	print(new_Y[i],new_X[i])
'''
acc,existence_acc,sensitivity, specifisity,NPV,PPV = acc_calc(final_decisions,[3,4,5],False)
print("After diagnosing laceration, accuracy of grade 3+4+5 is:",acc,", existence acc metric:",existence_acc,", sensitivity:",sensitivity,", specifisity:",specifisity,", NPV:",NPV,", PPV:",PPV)

acc,existence_acc,sensitivity, specifisity,NPV,PPV = acc_calc(final_decisions,[4,5],False)
print("After diagnosing laceration, accuracy of grade 4+5 is:",acc,", existence acc metric:",existence_acc,", sensitivity:",sensitivity,", specifisity:",specifisity,", NPV:",NPV,", PPV:",PPV)

acc,existence_acc,sensitivity, specifisity,NPV,PPV = acc_calc(final_decisions,[2],False)
print("After diagnosing laceration, accuracy of grade 2 is:",acc,", existence acc metric:",existence_acc,", sensitivity:",sensitivity,", specifisity:",specifisity,", NPV:",NPV,", PPV:",PPV)

ck,ck_weighted = cohen_kappa(final_decisions)
print("Cohen's kappa score is:",ck,", weighted Cohen's kappa score is:",ck_weighted)



# analyze AE
AE_csv = pd.read_csv("../UPDATED_spleen_project_dataset_deidentified_FINAL_KC_5.30.2021_-_updated_2.2.csv")
AE_data = AE_csv.values
AE_columns = AE_csv.columns.values
AE_column_dict = {}
for ind,column in enumerate(AE_columns):
	AE_column_dict[column] = int(ind)

for _data in AE_data[:,[AE_column_dict["patient ID"],AE_column_dict["AE_or_splnctmy"]]]:
		patient_name = 'spln inj '+str(_data[0])	
		final_decisions[patient_name]["AE"] = _data[1]


ae_auc = AE_auc(final_decisions)
print("AE auc is:",ae_auc)

# analyze laceration volume

lac_vol_csv = pd.read_csv("../splenic volume visualization and measurement.csv")
lac_vol_data = lac_vol_csv.values
lac_vol_columns = lac_vol_csv.columns.values
lac_vol_column_dict = {}
for ind,column in enumerate(lac_vol_columns):
	lac_vol_column_dict[column] = int(ind)

for _data in lac_vol_data[:,[lac_vol_column_dict["spleen inj anon case num"],lac_vol_column_dict["spln lac"]]]:
		patient_name = 'spln inj '+str(_data[0])	
		final_decisions[patient_name]["GT_lac_vol"] = _data[1]

lac_cor = pearson_score(final_decisions)
print("Laceration pearson coefficient is:",lac_cor)

RVD = relative_volume_difference(final_decisions)
print("Relative Volume Difference is:",RVD)

VS = volum_similarity(final_decisions)
print("Volume Similarity is:",VS)

lac_cor = pearson_score(final_decisions,phase=">1 cm^3")
print("Laceration pearson coefficient is:",lac_cor)

VS = volum_similarity(final_decisions,phase=">1 cm^3")
print("Volume Similarity is:",VS)

