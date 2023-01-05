python train.py --config-file ab_mip_newdata.yaml --num-gpus 1
cd output
mv model_final.pth model_final_fold0_ab_mip_newdata.pth
mv metrics.json metrics_fold0_ab_mip_newdata.json
mv log.txt log_fold_0_ab_mip_newdata.txt
mv final_result_fold0.json final_result_fold0_ab_mip_newdata.json
rm event*
cd ..

sed -i s/fold0/fold1/g ab_mip_newdata.yaml

python train.py --config-file ab_mip_newdata.yaml --num-gpus 1
cd output
mv model_final.pth model_final_fold1_ab_mip_newdata.pth
mv metrics.json metrics_fold1_ab_mip_newdata.json
mv log.txt log_fold_1_ab_mip_newdata.txt
mv final_result_fold1.json final_result_fold1_ab_mip_newdata.json
rm event*
cd ..

sed -i s/fold1/fold2/g ab_mip_newdata.yaml

python train.py --config-file ab_mip_newdata.yaml --num-gpus 1 
cd output
mv model_final.pth model_final_fold2_ab_mip_newdata.pth
mv metrics.json metrics_fold2_ab_mip_newdata.json
mv log.txt log_fold_2_ab_mip_newdata.txt
mv final_result_fold2.json final_result_fold2_ab_mip_newdata.json
rm event*
cd ..

sed -i s/fold2/fold3/g ab_mip_newdata.yaml

python train.py --config-file ab_mip_newdata.yaml --num-gpus 1
cd output
mv model_final.pth model_final_fold3_ab_mip_newdata.pth
mv metrics.json metrics_fold3_ab_mip_newdata.json
mv log.txt log_fold_3_ab_mip_newdata.txt
mv final_result_fold3.json final_result_fold3_ab_mip_newdata.json
rm event*
cd ..

sed -i s/fold3/fold4/g ab_mip_newdata.yaml

python train.py --config-file ab_mip_newdata.yaml --num-gpus 1
cd output
mv model_final.pth model_final_fold4_ab_mip_newdata.pth
mv metrics.json metrics_fold4_ab_mip_newdata.json
mv log.txt log_fold_4_ab_mip_newdata.txt
mv final_result_fold4.json final_result_fold4_ab_mip_newdata.json
rm event*
cd ..

sed -i s/fold4/fold0/g ab_mip_newdata.yaml
