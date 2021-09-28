# DARP: Distribution Aligning Refinery of Pseudo-label for Imbalanced Semi-supervised Learning

This repository contains code for the paper
**"Distribution Aligning Refinery of Pseudo-label for Imbalanced Semi-supervised Learning"** 
by [Jaehyung Kim](https://sites.google.com/view/jaehyungkim), Youngbum Hur, Sejun Park, Eunho Yang, Sung Ju Hwang, and Jinwoo Shin.

## Dependencies

* `python3`
* `pytorch == 1.1.0`
* `torchvision`
* `progress`
* `scipy`
* `randAugment (Pytorch re-implementation: https://github.com/ildoonet/pytorch-randaugment)`

## Scripts
Please check out `run.sh` for the scripts to run the baseline algorithms and ours (DARP).

### Training procedure of DARP 
Train a network with baseline algorithm, e.g., MixMatch on CIFAR-10
```
python train.py --gpu 0 --semi_method mix --dataset cifar10 --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 1 \
--epoch 500 --val-iteration 500
```
Applying DARP on the baseline algorithm
```
#python train.py --gpu 0 --darp --est --alpha 2 --warm 200 --semi_method mix --dataset cifar10 --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 1  \
--epoch 500 --val-iteration 500
```
