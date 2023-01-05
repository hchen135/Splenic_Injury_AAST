python train.py --config-file ab.yaml --num-gpus 1
cd output
mv model_final.pth model_final_fold0_ab.pth
mv metrics.json metrics_fold0_ab.json
mv log.txt log_fold_0_ab.txt
rm event*
cd ..

sed -i s/fold0/fold1/g ab.yaml

python train.py --config-file ab.yaml --num-gpus 1
cd output
mv model_final.pth model_final_fold1_ab.pth
mv metrics.json metrics_fold1_ab.json
mv log.txt log_fold_1_ab.txt
rm event*
cd ..

sed -i s/fold1/fold2/g ab.yaml

python train.py --config-file ab.yaml --num-gpus 1
cd output
mv model_final.pth model_final_fold2_ab.pth
mv metrics.json metrics_fold2_ab.json
mv log.txt log_fold_2_ab.txt
rm event*
cd ..

sed -i s/fold2/fold3/g ab.yaml

python train.py --config-file ab.yaml --num-gpus 1
cd output
mv model_final.pth model_final_fold3_ab.pth
mv metrics.json metrics_fold3_ab.json
mv log.txt log_fold_3_ab.txt
rm event*
cd ..

sed -i s/fold3/fold4/g ab.yaml

python train.py --config-file ab.yaml --num-gpus 1
cd output
mv model_final.pth model_final_fold4_ab.pth
mv metrics.json metrics_fold4_ab.json
mv log.txt log_fold_4_ab.txt
rm event*
cd ..

sed -i s/fold4/fold0/g ab.yaml
