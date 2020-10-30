########## Training ##########

## Baseline algorithm ##
# Table 1 
#python train_remix.py --align --gpu 0 --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 500 --val-iteration 500 --out cifar10@N_1500_r_150_remix
# Table 2
#python train_remix.py --est --align --gpu 0 --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 1 --epoch 500 --val-iteration 500 --out cifar10@N_1500_r_100_1_remix_est

## Applying DARP on the baseline algorithm ## 
# Table 1 
#python train_remix.py --darp --align --alpha 2 --warm 200 --gpu 0 --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 500 --val-iteration 500 --out cifar10@N_1500_r_150_remix_darp
# Table 2
#python train_remix.py --darp --est --align --alpha 2 --warm 200 --gpu 0 --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 1 --epoch 500 --val-iteration 500 --out cifar10@N_1500_r_100_1_remix_darp

########## Evaluation ##########
#python eval.py --pre_trained ./cifar10@N_1500_r_150_remix

########## Estimation ##########
#python train_base_estim.py --num_val 10 --gpu 0 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 1 --epoch 200 --val-iteration 200 --out cifar10@N_1500_r_100_1_estim
