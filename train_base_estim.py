from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

import models.wrn as models
import dataset.base_cifar10_split as dataset
from numpy.linalg import inv, pinv, cond
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='Weight Decay', help='weight decaying')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
parser.add_argument('--num_max', type=int, default=1500,
                        help='Number of labeled data')
parser.add_argument('--num_val', type=int, default=10,
                        help='Number of validation data per class')
parser.add_argument('--imb_ratio_l', type=int, default=100,
                        help='Number of labeled data')
parser.add_argument('--imb_ratio_u', type=int, default=100,
                        help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=80,
                        help='Number of labeled data')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
parser.add_argument('--alpha', default=0.0, type=float)
parser.add_argument('--th_cond', default=100, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing cifar10')
    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, 10, args.imb_ratio_l)
    U_SAMPLES_PER_CLASS = make_imb_data(2 * args.num_max, 10, args.imb_ratio_u)
    U_SAMPLES_PER_CLASS_T = torch.Tensor(U_SAMPLES_PER_CLASS)
    train_labeled_set, _, train_unlabeled_set, val_set, test_set = dataset.get_cifar10('/home/jaehyung/data',
                                                                                       N_SAMPLES_PER_CLASS,
                                                                                       U_SAMPLES_PER_CLASS,
                                                                                       args.num_val)
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                          drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                          drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    print("==> creating WRN-28-2")

    def create_model(ema=False):
        model = models.WRN(2, 10)
        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    ema_optimizer= WeightEMA(model, ema_model, alpha=args.ema_decay)
    start_epoch = 0

    # Resume
    title = 'imb-cifar-10'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Valid Loss', 'Valid Acc.', 'Test Loss', 'Test Acc.', 'Test GM.'])

    test_accs = []
    test_gms = []

    # Train and val
    n_infer = 0
    final_q = torch.zeros(10)
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()

    for epoch in range(start_epoch, args.epochs):
        # Adjust learning rate
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss = train(labeled_trainloader, model, optimizer, ema_optimizer, criterion, use_cuda)
        _, train_acc, train_cls, _, _ = validate(labeled_trainloader, model, epoch, use_cuda, mode='Train Stats')
        val_loss, val_acc, val_cls, val_gm, val_cls_acc = validate(val_loader, model, criterion, use_cuda, mode='Valid Stats')
        test_loss, test_acc, test_cls, test_gm, test_cls_acc = validate(test_loader, model, criterion, use_cuda, mode='Test Stats')

        # append logger file
        logger.append([train_loss, val_loss, val_acc, test_loss, test_acc, test_gm])

        # estimation of distribution for unlabeled data and save model
        est_q, cond_val = estimate_q_y(val_loader, unlabeled_trainloader, model, 10)
        is_zero = (est_q < 0).float().sum()
        is_best = (val_acc > best_acc) and (is_zero == 0)

        # no negative element and stable inverse
        if is_zero == 0 and cond_val < args.th_cond:
            print("===== Accepted (gap: {})=====".format((est_q - U_SAMPLES_PER_CLASS_T).abs().sum()))
            print(est_q)

            n_infer += 1
            final_q += est_q

        if is_best:
            best_acc = max(val_acc, best_acc)

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        test_accs.append(test_acc)
        test_gms.append(test_gm)

        # Save the best scores
        if is_best:
            best_bACC, best_GM = test_acc, test_gm
            best_major, best_neutral, best_minor = test_cls[0], test_cls[1], test_cls[2]

    logger.close()

    final_q = (final_q / n_infer)
    print(final_q)
    np.save(str(args.out) + '.npy', np.array(final_q))

    print('Best acc:')
    print(best_acc)

    print('Mean acc:')
    print(np.mean(test_accs[-20:]))

    print('Mean gm:')
    print(np.mean(test_gms[-20:]))

    print("Final Performance bACC : {}, GM : {}, major: {}, neutral : {}, minor: {}"
          .format(best_bACC, 100 * best_GM, 100 * best_major, 100 * best_neutral, 100 * best_minor))
    
def train(labeled_trainloader, model, optimizer, ema_optimizer, criterion, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)

    model.train()
    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets = labeled_train_iter.next()

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1,1), 1)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)

        all_inputs, all_targets = inputs_x, targets_x

        if args.alpha == 0:
            l = 1
        else:
            l = np.random.beta(args.alpha, args.alpha)
            l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        outputs, _ = model(mixed_input)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs, dim=1) * mixed_target, dim=1))
        loss = Lx

        # record loss
        losses.update(loss.item(), inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} '.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    )
        bar.next()
    bar.finish()

    return losses.avg

def validate(valloader, model, criterion, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    num_class = 10
    criterion = nn.CrossEntropyLoss()
    classwise_correct = torch.zeros(num_class)
    classwise_num = torch.zeros(num_class)
    section_acc = torch.zeros(3)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # classwise prediction
            pred_label = outputs.max(1)[1]
            pred_mask = (targets == pred_label).float()
            for i in range(num_class):
                class_mask = (targets == i).float()

                classwise_correct[i] += (class_mask * pred_mask).sum()
                classwise_num[i] += class_mask.sum()

             # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()

    # Major, Neutral, Minor
    section_num = int(num_class / 3)
    classwise_acc = (classwise_correct / classwise_num)
    section_acc[0] = classwise_acc[:section_num].mean()
    section_acc[2] = classwise_acc[-1 * section_num:].mean()
    section_acc[1] = classwise_acc[section_num:-1 * section_num].mean()

    GM = 1
    for i in range(10):
        if classwise_acc[i] == 0:
            # To prevent the N/A values, we set the minimum value as 0.001
            GM *= (1/1000) ** (1/10)
        else:
            GM *= (classwise_acc[i]) ** (1/10)

    return (losses.avg, top1.avg, section_acc.numpy(), GM, classwise_acc)


def make_imb_data(max_num, class_num, gamma):
    mu = np.power(1/gamma, 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / gamma))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    print(class_num_list)
    return list(class_num_list)

def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.00 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 160 and 180 epoch"""
    lr = args.lr

    if epoch >= 0.8 * args.epochs:
        lr /= 10
    if epoch >= 0.9 * args.epochs:
        lr /= 10

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def estimate_q_y(val_loader, u_loader, model, num_class):
    model.eval()

    conf_val = confusion(model, val_loader, num_class)
    conf_unl = confusion(model, u_loader, num_class)

    for i in range(num_class):
        conf_val[:, i] /= conf_val[:, i].sum()

    cond_val = cond(conf_val.numpy())
    print("Condition valude: {}".format(cond_val))

    inv_conf_val = torch.Tensor(inv(conf_val.numpy()))
    q_y_tilde = conf_unl.sum(1)
    q_y_esti = torch.matmul(inv_conf_val, q_y_tilde)

    return q_y_esti, cond_val

def confusion(model, loader, num_class):
    model.eval()

    num_classes = torch.zeros(num_class)
    confusion = torch.zeros(num_class, num_class)

    for batch_idx, (inputs, targets) in enumerate(loader):
        batch_size = inputs.size(0)
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs, _ = model(inputs)
        probs = torch.softmax(outputs.data, dim=1)

        # Update the confusion matrix
        for i in range(batch_size):
            confusion[:, targets[i]] += probs[i].cpu()
            num_classes[targets[i]] += 1

    return confusion

if __name__ == '__main__':
    main()
