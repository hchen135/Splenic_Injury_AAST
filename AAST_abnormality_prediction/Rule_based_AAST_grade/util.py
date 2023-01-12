import numpy as np
from sklearn.metrics import cohen_kappa_score,roc_auc_score
from scipy.stats import pearsonr

def acc_calc(result_dict,classes,print_bool = True):
	print("start accuracy calculation")
	pred = []
	GT = []
	patient_all = []
	for patient in result_dict:
		pred.append(result_dict[patient]["pred"])
		GT.append(result_dict[patient]["GT"])
		patient_all.append(patient)
	acc_list = []
	acc_existence_list = []
	for ind in range(len(pred)):
		if pred[ind] not in classes:
			pred[ind] = 0
		if GT[ind] not in classes:
			GT[ind] = 0
		# print("GT:",GT[ind],", pred:",pred[ind])
		acc_list.append(pred[ind] == GT[ind])
		acc_existence_list.append((pred[ind] > 0) == (GT[ind] > 0))
		if print_bool:
			if pred[ind] > 0 and GT[ind] > 0 and pred[ind] != GT[ind]:
				print("misclassified within classes: GT:",GT[ind],", pred:",pred[ind],", patient:",patient_all[ind])
			elif pred[ind] > 0 and GT[ind] == 0 :
				print("over-diagnosed: GT:",GT[ind],", pred:",pred[ind],", patient:",patient_all[ind])
			elif pred[ind] == 0 and GT[ind] > 0 :
				print("missed: GT:",GT[ind],", pred:",pred[ind],", patient:",patient_all[ind])
	if print_bool:
		return np.average(acc_list), np.average(acc_existence_list)
	else:
		TP = np.sum((np.array(pred) > 0)*(np.array(GT) > 0))
		TN = np.sum((np.array(pred) == 0)*(np.array(GT) == 0))
		FP = np.sum((np.array(pred) > 0)*(np.array(GT) == 0))
		FN = np.sum((np.array(pred) == 0)*(np.array(GT) > 0))
		print(TP,TN,FP,FN)
		sensitivity = TP / (TP + FN)
		specifisity = TN / (TN + FP)
		NPV = TN / (TN + FN)
		PPV = TP / (TP + FP)

		return np.average(acc_list), np.average(acc_existence_list),sensitivity,specifisity,NPV,PPV

def cohen_kappa(result_dict):
	pred = []
	GT = []
	for patient in result_dict:
		pred.append(result_dict[patient]["pred"])
		GT.append(result_dict[patient]["GT"])

	return cohen_kappa_score(pred,GT),cohen_kappa_score(pred,GT,weights='linear')

def AE_auc(result_dict):
	pred = []
	GT = []
	for patient in result_dict:
		pred.append(str(result_dict[patient]["pred"]))
		GT.append(result_dict[patient]["AE"])
	print(np.unique(GT))
	return roc_auc_score(GT,pred)

def pearson_score(result_dict,phase=""):
	pred = []
	GT = []
	for patient in result_dict:
		if result_dict[patient]["GT_ab"] == 0 and result_dict[patient]["GT_psa"] == 0:
			if (phase == ">1 cm^3" and result_dict[patient]["GT_lac_vol"] > 1) or (phase == "" and result_dict[patient]["GT_lac_vol"] > 0):
				pred.append(result_dict[patient]["pred_lac_vol"])
				GT.append(result_dict[patient]["GT_lac_vol"])
	return pearsonr(GT,pred)

def relative_volume_difference(result_dict,phase=""):
	metric = []
	for patient in result_dict:
		GT = result_dict[patient]["GT_lac_vol"]
		pred = result_dict[patient]["pred_lac_vol"]
		if result_dict[patient]["GT_ab"] == 0 and result_dict[patient]["GT_psa"] == 0:
			if (phase == ">1 cm^3" and GT > 1) or (phase == "" and GT > 0):
				metric.append(np.abs(GT-pred/1000)/GT)
				# print(patient,GT,pred/1000,np.abs(GT-pred/1000)/GT)
	# print(metric)
	return np.average(metric)

def volum_similarity(result_dict,phase=""):
	metric = []
	for patient in result_dict:
		GT = result_dict[patient]["GT_lac_vol"]
		pred = result_dict[patient]["pred_lac_vol"]
		if result_dict[patient]["GT_ab"] == 0 and result_dict[patient]["GT_psa"] == 0:
			if (phase == ">1 cm^3" and GT > 1) or (phase == "" and GT > 0):
				metric.append(1-np.abs(GT-pred/1000)/(GT+pred/1000))
				# print(patient,GT,pred/1000,1-np.abs(GT-pred/1000)/(GT+pred/1000))
	# print(metric)
	return np.average(metric)

