from __future__ import print_function

import argparse

parser = argparse.ArgumentParser(description='PyTorch MixMatch/ReMixMatch/FixMatch/DARP Evaluation')
# Checkpoints
parser.add_argument('--pre_trained', default='./pre_trained_path', type=str, help='path to log for pre-trained model')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

f = open(args.pre_trained + "/log.txt")
line = f.readlines()
bACC= 0
GM = 0
for i in range(1,21):
    if i > 1:
        bACC += float(line[-1 * i:-1 * (i-1)][0].split('\t')[-3])
        GM += float(line[-1 * i:-1 * (i-1)][0].split('\t')[-2])
    else:
        bACC += float(line[-1:][0].split('\t')[-3])
        GM += float(line[-1:][0].split('\t')[-2])
        
print(args.pre_trained)
print("Average bACC of last 20 epochs : {}".format(bACC/20))
print("Average GM of last 20 epochs : {}".format(GM/20))
