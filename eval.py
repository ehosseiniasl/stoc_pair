
from argparse import Namespace


import argparse
import time
import math
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import pickle as pkl

from models import ResNet_model, LinearNet

import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from utils import progress_bar, add_gaussian_noise, Cutout, _split_train_val, stoc_recontr_loss
from torch.optim import SGD
import collections
import torchvision.datasets as datasets
import glob
import tqdm
import torch.utils.data as utils
from utils import MNISTM



parser = argparse.ArgumentParser(description='Stochastic pairing')

# Directories
parser.add_argument('--data', type=str, default='datasets/',
                    help='location of the data corpus')
parser.add_argument('--root_dir', type=str, default='default/',
                    help='root dir path to save the log and the final model')
parser.add_argument('--save_dir', type=str, default='0/',
                    help='dir path (inside root_dir) to save the log and the final model')

# Hyperparams
parser.add_argument('--dataset', type=str, default='mnistm',
                    help='dataset name (cifar10)')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--bs', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--mbs', type=int, default=128, metavar='N',
                    help='minibatch size')


# meta specifications
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--cluster', action='store_true', help='do not show the progress bar for batch job')
parser.add_argument('--gpu', nargs='+', type=int, default=[0])


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in args.gpu)


args.root_dir = os.path.join('runs/', args.root_dir)
args.save_dir = os.path.join(args.root_dir, args.save_dir) 

use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

print('==> Preparing data..')
if args.dataset=='mnist':
    trans = ([transforms.ToTensor(),
                ]) # transforms.Normalize((0.5,), (1.0,))
    trans = transforms.Compose(trans)
    fulltrainset = torchvision.datasets.MNIST(root=args.data, train=True, transform=trans, download=True)

    train_set, valset = _split_train_val(fulltrainset, val_fraction=0.1)


    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.mbs, shuffle=True,
                                              num_workers=2, pin_memory=True)
    validloader = torch.utils.data.DataLoader(valset, batch_size=args.mbs, shuffle=False,
                                              num_workers=2, pin_memory=True)


    test_set = torchvision.datasets.MNIST(root=args.data, train=False, transform=trans)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=args.mbs, shuffle=False, num_workers=2)

    nb_classes = 10
    dim_inp=28*28 # np.prod(train_set.data.size()[1:])

elif args.dataset=='mnistm':
    trans = ([transforms.ToTensor(),
                ]) # transforms.Normalize((0.5,), (1.0,))
    trans = transforms.Compose(trans)
    fulltrainset = MNISTM(root=args.data, train=True, transform=trans, download=True)

    train_set, valset = _split_train_val(fulltrainset, val_fraction=0.1)


    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.mbs, shuffle=True,
                                              num_workers=2, pin_memory=True)
    validloader = torch.utils.data.DataLoader(valset, batch_size=args.mbs, shuffle=False,
                                              num_workers=2, pin_memory=True)


    test_set = MNISTM(root=args.data, train=False, transform=trans)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=args.mbs, shuffle=False, num_workers=2)

    nb_classes = 10
    dim_inp=28*28 # np.prod(train_set.data.size()[1:])
    




def test(epoch, loader, valid=False,train_loss=None, model=None):
    global best_acc, args
    model.eval()
    test_loss, correct, total = 0,0,0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = (model(inputs))
            _, predicted = torch.max(nn.Softmax(dim=1)(outputs).data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        if not args.cluster:
            progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    # Save checkpoint.
    acc = 100.*float(correct)/float(total)

    if valid and acc > best_acc:
        best_acc = acc
        print('Saving best model..')
        state = {
            'model': model,
            'epoch': epoch
        }
        with open(args.save_dir + '/best_model.pt', 'wb') as f:
            torch.save(state, f)

    return acc


with open(args.save_dir + '/best_model.pt', 'rb') as f:
    best_state = torch.load(f)
    model = best_state['model']
    if use_cuda:
    	model.cuda()
    # Run on test data.
    test_acc = test(0, testloader, model=model)
    best_val_acc = test(0, validloader, model=model)
    print('=' * 89)
    status = '| End of training | test acc {:3.4f} at best val acc {:3.4f}'.format(test_acc, best_val_acc)
    print(status)



