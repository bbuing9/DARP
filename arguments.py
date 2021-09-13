import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch DARP Training')

    # Optimization options
    parser.add_argument('--semi_method', default='mix', help='Semi: mix | remix | fix')
    parser.add_argument('--dataset', default='cifar10', help='Dataset: cifar10 | cifar100 | stl10')
    parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.002, type=float, metavar='LR', help='initial learning rate')
    # Checkpoints
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--out', default='result', help='Directory to output the result')
    # Miscs
    parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
    # Device options
    parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

    # Method options
    parser.add_argument('--num_max', type=int, default=1500, help='Number of samples in the maximal class')
    parser.add_argument('--ratio', type=float, default=2.0, help='Relative size between labeled and unlabeled data')
    parser.add_argument('--imb_ratio_l', type=int, default=100, help='Imbalance ratio for labeled data')
    parser.add_argument('--imb_ratio_u', type=int, default=100, help='Imbalance ratio for unlabeled data')
    parser.add_argument('--step', action='store_true', help='Type of class-imbalance')
    parser.add_argument('--val-iteration', type=int, default=500, help='Frequency for the evaluation')

    # Common Hyper-parameters for Semi-supervised Methods
    parser.add_argument('--mix_alpha', default=0.75, type=float)
    parser.add_argument('--lambda-u', default=75, type=float)
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--ema-decay', default=0.999, type=float)

    # Hyper-parameters for ReMixMatch
    parser.add_argument('--w_rot', default=0.5, type=float)
    parser.add_argument('--w_ent', default=0.5, type=float)
    parser.add_argument('--align', action='store_true', help='Distribution alignment term')

    # Hyperparameters for FixMatch
    parser.add_argument('--tau', default=0.95, type=float, help='hyper-parameter for pseudo-label of FixMatch')

    # Hyperparameters for DARP
    parser.add_argument('--warm', type=int, default=500, help='Number of warm up epoch for DARP')
    parser.add_argument('--alpha', default=2.0, type=float, help='hyperparameter for removing noisy entries')
    parser.add_argument('--darp', action='store_true', help='Applying DARP')
    parser.add_argument('--est', action='store_true', help='Using estimated distribution for unlabeled dataset')
    parser.add_argument('--iter_T', type=int, default=10, help='Number of iteration (T) for DARP')
    parser.add_argument('--num_iter', type=int, default=10, help='Scheduling for updating pseudo-labels')

    return parser.parse_args()