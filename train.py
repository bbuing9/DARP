# This code is constructed based on Pytorch Implementation of MixMatch(https://github.com/YU1ut/MixMatch-pytorch)

from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from scipy import optimize

import models.wrn as models
from arguments import parse_args
from dataset import get_cifar10, get_cifar100, get_stl10
from training_functions import trains
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from common import validate, estimate_pseudo, opt_solver, make_imb_data, save_checkpoint, SemiLoss, WeightEMA, interleave

args = parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0  # best test accuracy
if args.dataset == 'cifar100':
    args.num_class = 100
else:
    args.num_class = 10

if args.semi_method == 'remix':
    args.lambda_u = 1.5

def main():
    global best_acc

    args.out = args.dataset + '@N_' + str(args.num_max) + '_r_'
    if args.imb_ratio_l == args.imb_ratio_u:
        args.out += str(args.imb_ratio_l) + '_' + args.semi_method
    else:
        args.out += str(args.imb_ratio_l) + '_' + str(args.imb_ratio_u) + '_' + args.semi_method

    if args.darp:
        args.out += '_darp_alpha' + str(args.alpha) + '_iterT' + str(args.iter_T)

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, args.num_class, args.imb_ratio_l)
    U_SAMPLES_PER_CLASS = make_imb_data(args.ratio * args.num_max, args.num_class, args.imb_ratio_u)
    N_SAMPLES_PER_CLASS_T = torch.Tensor(N_SAMPLES_PER_CLASS)

    print(args.out)

    if args.dataset == 'cifar10':
        print(f'==> Preparing imbalanced CIFAR-10')
        train_labeled_set, train_unlabeled_set, test_set = get_cifar10('/home/jaehyung/data', N_SAMPLES_PER_CLASS,
                                                                               U_SAMPLES_PER_CLASS, args.out)
    elif args.dataset == 'stl10':
        print(f'==> Preparing imbalanced STL-10')
        train_labeled_set, train_unlabeled_set, test_set = get_stl10('/home/jaehyung/data', N_SAMPLES_PER_CLASS, args.out)
    elif args.dataset == 'cifar100':
        print(f'==> Preparing imbalanced CIFAR-100')
        train_labeled_set, train_unlabeled_set, test_set = get_cifar100('/home/jaehyung/data', N_SAMPLES_PER_CLASS,
                                                                                U_SAMPLES_PER_CLASS, args.out)
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                          drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                            drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    print("==> creating WRN-28-2")
    def create_model(ema=False):
        model = models.WRN(2, args.num_class)
        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ema_optimizer= WeightEMA(model, ema_model, lr=args.lr, alpha=args.ema_decay)
    start_epoch = 0

    # Resume
    title = 'Imbalanced' + '-' + args.dataset + '-' + args.semi_method
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U', 'Test Loss', 'Test Acc.', 'Test GM.'])

    test_accs = []
    test_gms = []

    # Default values for MixMatch and DARP
    emp_distb_u = torch.ones(args.num_class) / args.num_class
    pseudo_orig = torch.ones(len(train_unlabeled_set.data), args.num_class) / args.num_class
    pseudo_refine = torch.ones(len(train_unlabeled_set.data), args.num_class) / args.num_class

    # Main function
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        # Use the estimated distribution of unlabeled data
        if args.est:
            if args.dataset == 'cifar10':
                est_name = './estimation/cifar10@N_1500_r_{}_{}_estim.npy'.format(args.imb_ratio_l, args.imb_ratio_u)
            else:
                est_name = './estimation/stl10@N_450_r_{}_estim.npy'.format(args.imb_ratio_l)
            est_disb = np.load(est_name)
            target_disb = len(train_unlabeled_set.data) * torch.Tensor(est_disb) / np.sum(est_disb)
        # Use the inferred distribution with labeled data
        else:
            target_disb = N_SAMPLES_PER_CLASS_T * len(train_unlabeled_set.data) / sum(N_SAMPLES_PER_CLASS)
        
        train_loss, train_loss_x, train_loss_u, emp_distb_u, pseudo_orig, pseudo_refine = trains(args, labeled_trainloader,
                                                                                                unlabeled_trainloader,
                                                                                                model, optimizer,
                                                                                                ema_optimizer,
                                                                                                train_criterion,
                                                                                                epoch, use_cuda,
                                                                                                target_disb, emp_distb_u,
                                                                                                pseudo_orig, pseudo_refine)

        # Evaluation part
        test_loss, test_acc, test_cls, test_gm = validate(test_loader, ema_model, criterion, use_cuda,
                                                          mode='Test Stats', num_class=args.num_class)

        # Append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, test_loss, test_acc, test_gm])

        # Save models
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch + 1, args.out)
        test_accs.append(test_acc)
        test_gms.append(test_gm)

    logger.close()

    # Print the final results
    print('Mean bAcc:')
    print(np.mean(test_accs[-20:]))

    print('Mean GM:')
    print(np.mean(test_gms[-20:]))

    print('Name of saved folder:')
    print(args.out)

if __name__ == '__main__':
    main()
