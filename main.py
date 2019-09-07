'''
Stochastic Pairing

'''

from argparse import Namespace


import argparse
import time
import math
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import autograd
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
import json

parser = argparse.ArgumentParser(description='Stochastic Pairing')

# Directories
parser.add_argument('--data', type=str, default='datasets/',
                    help='location of the data corpus')
parser.add_argument('--root_dir', type=str, default='default/',
                    help='root dir path to save the log and the final model')
parser.add_argument('--save_dir', type=str, default='0/',
                    help='dir path (inside root_dir) to save the log and the final model')

# Hyperparams
parser.add_argument('--dataset', type=str, default='mnist',
                    help='dataset name (cifar10)')
parser.add_argument('--arch', type=str, default='mnist_cnn',
                    help='arch name (resnet, vgg11)')
parser.add_argument('--depth', type=int, default=56,
                    help='number of resblocks if using resnet architecture')
parser.add_argument('--k', type=int, default=1,
                    help='widening factor for wide resnet architecture')
parser.add_argument('--nb_filters', type=int, default=16,
                    help='number of base filters in resnet architecture')
parser.add_argument('--kernel', type=int, default=3,
                    help='kernel size for resnet architecture')


# stochastic pairing
parser.add_argument('--stoc_pair', action='store_true',
                    help='use stochastic pairing')
parser.add_argument('--beta', type=float, default=1,
                    help='coefficient of stochastic pairing')
parser.add_argument('--ret_hid', type=int, nargs='+', default=[0],
                    help='apply stoc pair to this layer ID')
parser.add_argument('--sp_spool', type=int, default=0,
                    help='pool repr with this pool size before applying stoc pair')
parser.add_argument('--sp_cpool', action='store_true',
                    help='max pool channelwise before applying stoc pair')


parser.add_argument('--bn_eval', action='store_true',
                    help='adapt BN stats during eval')


parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

# Optimization hyper-parameters
parser.add_argument('--opt', type=str, default="adam",
                    help='optimizer(sgd, adam)')
parser.add_argument('--bs', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--mbs', type=int, default=128, metavar='N',
                    help='minibatch size')
parser.add_argument('--nesterov', type=bool, default=False,
                    help='Use nesterov momentum T/F')
parser.add_argument('--normalization', type=str, default='bn',
                    help='type of normalization (wn, bn)')
parser.add_argument('--noaffine',  action='store_true',
                    help='no affine transformations')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate ')
parser.add_argument('--m', type=float, default=0.9,
                    help='momentum')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--init', type=str, default="he")
parser.add_argument('--wdecay', type=float, default=0.0001,
                    help='weight decay applied to all weights')
parser.add_argument('--sch', type=str, default='const',
                    help='LR schedule')


parser.add_argument('--cutout', action='store_true',
                    help='use cutout')
parser.add_argument('--dropout', type=float, default=0.,
                    help='dropout')


# meta specifications
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--cluster', action='store_true', help='do not show the progress bar for batch job')
parser.add_argument('--gpu', nargs='+', type=int, default=[0])


args = parser.parse_args()
args.root_dir = os.path.join('runs/', args.root_dir)
args.save_dir = os.path.join(args.root_dir, args.save_dir) 
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
log_dir = args.save_dir + '/'

with open(args.save_dir + '/config.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in args.gpu)


# if args.arch == 'mlp':
if args.sch=='sch1':
    lr_sch = [[40, 1], [60, 0.5], [90, 0.25], [1000000000000, 0.1]]
    mom_sch = [[99999999, 1.]]
elif args.sch=='sch2':
    lr_sch = [[90, 1], [136, 0.5], [175, 0.25], [1000000000000, 0.1]]
elif args.sch=='sch3':
    lr_sch = [[75, 1], [100, 0.5], [1000000000000, 0.25]]
    mom_sch = [[99999999, 1.]]
elif args.sch=='const':
    lr_sch = [[np.inf, 1.]]





# Set the random seed manually for reproducibility.
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)


###############################################################################
# Load data
###############################################################################

print('==> Preparing data..')
if args.dataset=='mnist':
    trans = ([transforms.ToTensor()]) 
    if args.cutout:
        trans += [Cutout(n_holes=1, length=8)]
    trans = transforms.Compose(trans)
    fulltrainset = torchvision.datasets.MNIST(root=args.data, train=True, transform=trans, download=True)

    train_set, valset = _split_train_val(fulltrainset, val_fraction=0.1)


    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.mbs, shuffle=True,
                                              num_workers=0, pin_memory=True)
    validloader = torch.utils.data.DataLoader(valset, batch_size=args.mbs, shuffle=False,
                                              num_workers=0, pin_memory=True)


    test_set = torchvision.datasets.MNIST(root=args.data, train=False, transform=trans)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=args.mbs, shuffle=False, num_workers=0)

    nb_classes = 10
    dim_inp=28*28 # np.prod(train_set.data.size()[1:])
elif 'cmnist' in args.dataset:
    data_dir_cmnist = args.data + 'cmnist/' + args.dataset + '/'
    data_x = np.load(data_dir_cmnist+'train_x.npy')
    data_y = np.load(data_dir_cmnist+'train_y.npy')
    # data_x = np.transpose(data_x, (0,3,1,2))
    data_x = torch.from_numpy(data_x).type('torch.FloatTensor')
    data_y = torch.from_numpy(data_y).type('torch.LongTensor')
    # print(data_x.size(), data_y.size())
    my_dataset = utils.TensorDataset(data_x,data_y)

    train_set, valset = _split_train_val(my_dataset, val_fraction=0.1)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.mbs, shuffle=True, num_workers=0)
    validloader = torch.utils.data.DataLoader(valset, batch_size=args.mbs, shuffle=False,
                                              num_workers=0, pin_memory=True)


    data_x = np.load(data_dir_cmnist+'test_x.npy')
    data_y = np.load(data_dir_cmnist+'test_y.npy')
    data_x = torch.from_numpy(data_x).type('torch.FloatTensor')
    data_y = torch.from_numpy(data_y).type('torch.LongTensor')
    my_dataset = utils.TensorDataset(data_x,data_y)
    testloader = torch.utils.data.DataLoader(my_dataset, batch_size=args.bs, shuffle=False, num_workers=0)

    nb_classes = 10
    dim_inp=28*28* 3
elif 'nico-a' in args.dataset:
    data_dir_nico_a = args.data + 'nico_animals_np/'
    data_x = np.load(data_dir_nico_a+'train_x.npy')
    data_y = np.load(data_dir_nico_a+'train_y.npy')
    # data_x = np.transpose(data_x, (0,3,1,2))
    data_x = torch.from_numpy(data_x).type('torch.FloatTensor')
    data_y = torch.from_numpy(data_y).type('torch.LongTensor')
    # print(data_x.size(), data_y.size())
    my_dataset = utils.TensorDataset(data_x,data_y)


    train_set, valset = _split_train_val(my_dataset, val_fraction=0.1)

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.mbs, shuffle=True, num_workers=0)
    validloader = torch.utils.data.DataLoader(valset, batch_size=args.mbs, shuffle=False,
                                              num_workers=1, pin_memory=True)


    data_x = np.load(data_dir_nico_a+'test_x.npy')
    data_y = np.load(data_dir_nico_a+'test_y.npy')
    # data_x = np.transpose(data_x, (0,3,1,2))
    data_x = torch.from_numpy(data_x).type('torch.FloatTensor')
    data_y = torch.from_numpy(data_y).type('torch.LongTensor')
    my_dataset = utils.TensorDataset(data_x,data_y)
    testloader = torch.utils.data.DataLoader(my_dataset, batch_size=args.mbs, shuffle=False, num_workers=0)

    nb_classes = 10
    dim_inp=256*256* 3
elif args.dataset=='cifar10':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    fulltrainset = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=transform_train)

    train_set, valset = _split_train_val(fulltrainset, val_fraction=0.1)


    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.mbs, shuffle=True,
                                              num_workers=0, pin_memory=True)
    validloader = torch.utils.data.DataLoader(valset, batch_size=args.mbs, shuffle=False,
                                              num_workers=0, pin_memory=True)


    testset = torchvision.datasets.CIFAR10(root=args.data, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.mbs, shuffle=False, num_workers=0)


    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    nb_classes = len(classes)
    dim_inp = 32*32*3
###############################################################################
# Build the model
###############################################################################


inp_channels = 1 if args.dataset=='mnist' else 3
print('==> Building model..')
start_epoch=0
if args.arch == 'linear':
    model = LinearNet(dim_inp, nb_classes)
elif args.arch == 'resnet':
    model = ResNet_model(dropout=args.dropout, normalization= args.normalization, num_classes=nb_classes, dataset=args.dataset, depth=args.depth, nb_filters=args.nb_filters, kernel_size=args.kernel,\
                        inp_channels=inp_channels, k=args.k, affine=not args.noaffine)

params = list(model.parameters())
model = torch.nn.DataParallel(model, device_ids=range(len(args.gpu)))


nb = 0
if args.init == 'he':
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nb += 1
            print ('Update init of ', m)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d) and not args.noaffine:
            print ('Update init of ', m)
            m.weight.data.fill_(1)
            m.bias.data.zero_()
print( 'Number of Conv layers: ', (nb))



if use_cuda:
    model.cuda()
total_params = sum(np.prod(x.size()) if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
print('Args:', args)
print( 'Model total parameters:', total_params)
with open(args.save_dir + '/log.txt', 'w') as f:
    f.write(str(args) + ',total_params=' + str(total_params) + '\n')

criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()


###############################################################################
# Training code
###############################################################################


def test(epoch, loader, valid=False,train_loss=None, model=None):
    global best_acc, args

    if args.bn_eval:
        model.train()
        for _ in range(2):
            for batch_idx, (inputs, targets) in enumerate(loader):
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                inputs, targets = Variable(inputs), Variable(targets)
                _ = (model(inputs))

    model.eval()
    test_loss, correct, total = 0,0,0
    for batch_idx, (inputs, targets) in enumerate(loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = (model(inputs))
            loss = criterion(outputs, targets)


            test_loss += loss.data
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

global_iters=0
def train(epoch):
    global trainloader
    global optimizer
    global args, params, ind, train_set
    global model, global_iters, best_loss
    # Turn on training mode which enables dropout.
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    regularization_loss = 0
    loss_list, regularization_loss_list = [], []
    tot_regularization_loss = 0
    total_loss_list=[]

    for lr_ in lr_sch:
        if epoch<= lr_[0]:
            lr = lr_[1]
            break

    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr*lr

    optimizer.zero_grad()
    if not hasattr(train, 'nb_samples_seen'):
        train.nb_samples_seen = 0

    

    iters=0
    tot_iters = len(trainloader)
    for batch_idx in range(tot_iters):
        inputs, targets = next(iter(trainloader)) 
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() 

        inputs = Variable(inputs)

        if args.stoc_pair:
            outputs, hid_repr = (model(inputs, args.ret_hid))
            # print(feat_dim, hid_repr.size())
        else:
            outputs = (model(inputs))

        loss = criterion(outputs, targets)

        if args.stoc_pair:
            regularization_loss = stoc_recontr_loss(epoch, hid_repr, hid_repr, targets.cpu().numpy(), spool=args.sp_spool, cpool=args.sp_cpool)
            tot_regularization_loss += regularization_loss.data

        total_loss_ = loss +  (args.beta)*regularization_loss
        total_loss_.backward() # retain_graph=True
        total_loss_list.append(loss.data.cpu().numpy() )
                                                                                                                                                                                                                                                                                                                                                                                                                       

        train_loss += loss.data
        _, predicted = torch.max(nn.Softmax(dim=1)(outputs).data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if train.nb_samples_seen+args.mbs==args.bs:
            global_iters+=1
            iters+=1
            
            for name, variable in model.named_parameters():
                g = variable.grad.data
                g.mul_(1./(1+train.nb_samples_seen/float(args.mbs)))


            train.nb_samples_seen = 0


            optimizer.step()

            optimizer.zero_grad()

            loss_list.append(sum(total_loss_list)/len(total_loss_list))
            total_loss_list=[]

        else:
            train.nb_samples_seen += args.mbs


        

        if not args.cluster:
            progress_bar(batch_idx, tot_iters, 'Epoch {:3d} | Loss: {:3f} | Acc: {:3f} | reg_loss {:3f} | LR {}'
                         .format (epoch, train_loss/(batch_idx+1) , 100.*float(correct)/(1e-7+float(total)), tot_regularization_loss/(batch_idx+1), args.lr* lr))

    loss = sum(loss_list)/float(len(loss_list))
    acc = 100.*correct/total
    regularization_loss = tot_regularization_loss/tot_iters
    return loss, acc, regularization_loss


if args.opt=='adam':
    optimizer = torch.optim.Adam(params,\
                                   lr=args.lr, weight_decay=args.wdecay)
elif args.opt=='sgd':
    optimizer = SGD(params, lr=args.lr*lr_sch[0][1], momentum=args.m, \
                    weight_decay=args.wdecay, nesterov=args.nesterov)

best_acc, best_loss =0, np.inf
lr_list, train_loss_list, train_acc_list, valid_acc_list, reg_loss_list, best_loss = [], [], [], [], [], float('inf')
epoch = start_epoch


def train_fn():
    global epoch, args, best_loss, best_acc
    while epoch<args.epochs:
        epoch+=1

        epoch_start_time = time.time()

        loss, train_acc, regularization_loss= train(epoch)

        

        train_loss_list.append(loss)
        train_acc_list.append(train_acc)
        reg_loss_list.append(regularization_loss)


        valid_acc = test(epoch, testloader, valid=True, train_loss=loss, model=model)
       
        valid_acc_list.append(valid_acc)

        with open(args.save_dir + "/train_loss.pkl", "wb") as f:
            pkl.dump(train_loss_list, f)

        with open(args.save_dir + "/train_acc.pkl", "wb") as f:
            pkl.dump(train_acc_list, f)

        with open(args.save_dir + "/valid_acc.pkl", "wb") as f:
            pkl.dump(valid_acc_list, f)

        with open(args.save_dir + "/reg_loss_list.pkl", "wb") as f:
            pkl.dump(reg_loss_list, f)

        lr_list.append(optimizer.param_groups[0]['lr'])


        status = 'Epoch {}/{} | Loss {:3.4f} | acc {:3.2f} | val-acc {:3.2f} | reg_loss : {:3.4f} | beta {:4.4f} | LR {:4.4f}'.\
            format( epoch,args.epochs, loss, train_acc, valid_acc, reg_loss_list[-1], args.beta, lr_list[-1])
        print (status)

        with open(args.save_dir + '/log.txt', 'a') as f:
            f.write(status + '\n')

        print('-' * 89)

train_fn()
status = '| End of training | best test acc {:3.4f} '.format(best_acc)
print(status)
with open(args.save_dir + '/log.txt', 'a') as f:
        f.write(status + '\n')


# def test_fn():
#     global args
#     with open(args.save_dir + '/best_model.pt', 'rb') as f:
#         best_state = torch.load(f)
#     model = best_state['model']
#     # Run on test data.
#     test_acc = test(epoch, validloader, model=model)
#     best_val_acc = test(epoch, testloader, model=model)
#     print('=' * 89)
#     status = '| End of training | test acc {:3.4f} at best val acc {:3.4f}'.format(test_acc, best_val_acc)
#     print(status)
#     with open(args.save_dir + '/log.txt', 'a') as f:
#             f.write(status + '\n')
# test_fn()