########## Training ##########

## Baseline algorithm ##
# Table 1 
#python train.py --semi_method remix --gpu 0 --dataset cifar10 --align --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 500 --val-iteration 500
# Table 2
#python train.py --semi_method remix --gpu 0 --dataset cifar10 --est --align --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 1 --epoch 500 --val-iteration 500

## Applying DARP on the baseline algorithm ## 
# Table 1 
#python train.py --semi_method remix --gpu 0 --dataset cifar10 --darp --align --alpha 2 --warm 200 --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 500 --val-iteration 500
# Table 2
#python train.py --semi_method remix --gpu 0 --dataset cifar10 --darp --est --align --alpha 2 --warm 200 --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 1 --epoch 500 --val-iteration 500

## CIFAR-100 & STL-10
# Original
#python train.py --semi_method mix --gpu 0 --dataset cifar100 --ratio 0.5 --num_max 300 --imb_ratio_l 20 --imb_ratio_u 20 --epoch 500 --val-iteration 500
#python train.py --semi_method mix --gpu 0 --dataset stl10 --num_max 450 --imb_ratio_l 20 --imb_ratio_u 1 --epoch 500 --val-iteration 500
# DARP
#python train.py --semi_method mix --gpu 0 --dataset cifar100 --darp --num_iter 100 --iter_T 20 --warm 200 --ratio 0.5 --num_max 300 --imb_ratio_l 20 --imb_ratio_u 20 --epoch 500 --val-iteration 500
#python train.py --semi_method mix --gpu 0 --dataset stl10 --est --darp --num_iter 100 --iter_T 20 --warm 200 --num_max 450 --imb_ratio_l 20 --imb_ratio_u 1 --epoch 500 --val-iteration 500

########## Evaluation ##########
#python eval.py --pre_trained ./cifar10@N_1500_r_150_remix

########## Estimation ##########
#python train_base_estim.py --num_val 10 --gpu 0 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 1 --epoch 200 --val-iteration 200 --out cifar10@N_1500_r_100_1_estim
