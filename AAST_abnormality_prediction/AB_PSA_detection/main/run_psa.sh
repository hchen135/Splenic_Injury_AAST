python train.py --config-file psa.yaml --num-gpus 1
cd output
mv model_final.pth model_final_fold0_psa.pth
mv metrics.json metrics_fold0_psa.json
mv log.txt log_fold_0_psa.txt
rm event*
cd ..

sed -i s/fold0/fold1/g psa.yaml

python train.py --config-file psa.yaml --num-gpus 1
cd output
mv model_final.pth model_final_fold1_psa.pth
mv metrics.json metrics_fold1_psa.json
mv log.txt log_fold_1_psa.txt
rm event*
cd ..

sed -i s/fold1/fold2/g psa.yaml

python train.py --config-file psa.yaml --num-gpus 1
cd output
mv model_final.pth model_final_fold2_psa.pth
mv metrics.json metrics_fold2_psa.json
mv log.txt log_fold_2_psa.txt
rm event*
cd ..

sed -i s/fold2/fold3/g psa.yaml

python train.py --config-file psa.yaml --num-gpus 1
cd output
mv model_final.pth model_final_fold3_psa.pth
mv metrics.json metrics_fold3_psa.json
mv log.txt log_fold_3_psa.txt
rm event*
cd ..

sed -i s/fold3/fold4/g psa.yaml

python train.py --config-file psa.yaml --num-gpus 1
cd output
mv model_final.pth model_final_fold4_psa.pth
mv metrics.json metrics_fold4_psa.json
mv log.txt log_fold_4_psa.txt
rm event*
cd ..

sed -i s/fold4/fold0/g psa.yaml
