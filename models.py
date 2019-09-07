
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import math

import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
from torch.nn.utils import weight_norm as WeightNorm
from torch.nn.init import orthogonal_, kaiming_normal_
import numpy as np


ACT = F.relu
# ACT = torch.sigmoid

class MLPLayer(nn.Module):
    def __init__(self, dim_in=None, dim_out=None, bn='bn', act=True, dropout=0.):
        super(MLPLayer, self).__init__()
        self.dropout = dropout
        self.act=act
        if bn=='wn':
            self.layer = [nn.utils.weight_norm(nn.Linear(dim_in, dim_out))]
        elif bn=='bn':
            fc = nn.Linear(dim_in, dim_out)
            bn_ = nn.BatchNorm1d(dim_out)
            self.layer = [fc, bn_]
        else:
            self.layer = [nn.Linear(dim_in, dim_out)]
        self.layer = nn.Sequential(*self.layer)
        
    def forward(self, x, slope=0):
        x=self.layer(x)
        if self.act:
            x = ACT(self.layer(x))
            # x = F.leaky_relu(x, slope)
            if self.dropout>0:
                x = nn.Dropout(self.dropout)(x)
#         else:
#             x = self.layer(x)
        return x



class LinearNet(nn.Module):
    def __init__(self, dim_in=None, dim_out=None):
        super(LinearNet, self).__init__()
        self.layer = nn.Linear(dim_in, dim_out)
        
    def forward(self, x, slope=0):
        x = x.view(x.size(0), -1)
        x=self.layer(x)
        return x


class resblock(nn.Module):

    def __init__(self, depth, channels, stride=1, dropout=0., normalization='', nresblocks=1.,affine=True, kernel_size=3):
        self.bn = normalization=='bn'
        self.depth = depth
        self. channels = channels
        
        super(resblock, self).__init__()
        self.bn1 = nn.BatchNorm2d(depth,affine=affine) if normalization=='bn' else nn.Sequential()
        self.conv2 = (nn.Conv2d(depth, channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False))
        self.conv2 = WeightNorm(self.conv2) if normalization=='wn' else self.conv2
        self.bn2 = nn.BatchNorm2d(channels, affine=affine) if normalization=='bn' else nn.Sequential()
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.conv3 = WeightNorm(self.conv3) if normalization=='wn' else self.conv3
        self.bn3 = nn.Sequential() # nn.BatchNorm2d(channels,affine=affine) if normalization=='bn' else nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride > 1 or depth!=channels:
            layers = []
            layers += [nn.Conv2d(depth, channels, kernel_size=1, stride=stride, padding=0, bias=False), nn.BatchNorm2d(channels,affine=affine) if normalization=='bn' else nn.Sequential()]
            self.shortcut = nn.Sequential(*layers)
            
        self.dropout = dropout

    def forward(self, x):
#         print 'input shape: ', x.size()
#         print 'depth, channels: ', self.depth, self.channels
        out = ACT(self.bn1(x))
        out = ACT(self.bn2(self.conv2(out)))
        
        out = self.bn3(self.conv3(out))
        
#         print 'output shapes: ', out.size(), self.shortcut(x).size()
        short = self.shortcut(x)
        # print(out.size(), short.size())
        out += 1.*short
        return out



class ResNet(nn.Module):
    def __init__(self, depth=56, nb_filters=16, num_classes=10, dropout=0., normalization='', dataset=None, affine=True, kernel_size=3, inp_channels=3, k=1, pad_conv1=0): # n=9->Resnet-56
        super(ResNet, self).__init__()
        nstage = 3 # FIXME 3
        self.dataset=dataset
        
        self.pre_clf=[]

        assert ((depth-2)%6 ==0), 'resnet depth should be 6n+2'
        n = int((depth-2)/6)
        
        nfilters = [nb_filters*k, nb_filters*k, 2* nb_filters*k, 4* nb_filters*k, num_classes]
        self.nfilters = nfilters
        self.num_classes = num_classes
        self.conv1 = (nn.Conv2d(inp_channels, nfilters[0], kernel_size=kernel_size, stride=1, padding=pad_conv1, bias=False))
        self.conv1 = WeightNorm(self.conv1) if normalization=='wn' else self.conv1
        # self.layers.append(conv1)
        self.bn1 = nn.BatchNorm2d(nb_filters*k, affine=affine) if normalization=='bn' else nn.Sequential()
        # self.layers.append(layer)


        nb_filters_prev = nb_filters_cur = nb_filters*k
        for stage in range(nstage):
            self.layers = []
            nb_filters_cur =  nfilters[stage+1] # (2 ** stage) * nb_filters* k
            for i in range(n):
                subsample = 1 if (i > 0 or stage == 0) else 2
                layer = resblock(nb_filters_prev, nb_filters_cur, subsample, dropout=dropout, normalization=normalization, nresblocks = nstage*n, affine=affine, kernel_size=3)
                self.layers.append(layer)
                nb_filters_prev = nb_filters_cur
            self.pre_clf.append(nn.Sequential(*self.layers))

            self.pre_clf = nn.ModuleList(self.pre_clf)
        
        
        
        # self.pre_clf = nn.Sequential(*self.layers)

        self.feat_dim = nb_filters_cur#*8*8

        # self.bn_fc = nn.BatchNorm1d(nb_filters_cur, affine=affine) if normalization=='bn' else nn.Sequential()

        self.fc = MLPLayer(nb_filters_cur, nfilters[-1], 'none', act=False)
        
        
        
    def forward(self, x, ret_hid=[], ret_out=True):
        if x.size()[1]==1:
            out = torch.ones(x.size(0), 3, x.size(2), x.size(3)).type('torch.cuda.FloatTensor')
            out = out*x
        else:
            out = x
#         for layer in self.layers:
#             out = layer(out)

        hid_list=[]
        hid = self.conv1(out)
        if 0 in ret_hid and not ret_out:
            return hid
        hid_list.append(hid)
        hid = self.bn1(hid)
        
        # out = ACT(out)




        hid0 = self.pre_clf[0](hid)

        hid0_ = self.pre_clf[0](hid.detach())
        hid1 = self.pre_clf[1](hid0)
        hid1_ = self.pre_clf[1](hid0.detach())
        hid2 = self.pre_clf[2](hid1)
        hid2_ = self.pre_clf[2](hid1.detach())
        hid_list.extend([hid0_,hid1_,hid2_])


        fc = torch.mean(hid2.view(hid2.size(0), hid2.size(1), -1), dim=2)

        fc = fc.view(fc.size()[0], -1)


        out = self.fc((fc))
        out_ = self.fc((fc.detach()))


        hid_list.append(out_)


        if ret_hid!=[]:
            return out, [hid_list[i] for i in ret_hid]
        else:
            return out



# Resnet nomenclature: 6n+2 = 3x2xn + 2; 3 stages, each with n number of resblocks containing 2 conv layers each, and finally 2 non-res conv layers
def ResNet56(dropout=0., normalization='bn', num_classes=10, dataset='cifar10'):
    return ResNet(n=9, nb_filters=16, num_classes=num_classes, dropout=dropout, normalization=normalization, dataset=dataset)
def ResNet_model(dropout=0., normalization='bn', num_classes=10, dataset='cifar10', depth=56, nb_filters=16, kernel_size=3, inp_channels=3, k=1, pad_conv1=0, affine=True):
    return ResNet(depth=depth, nb_filters=nb_filters, num_classes=num_classes, dropout=dropout, normalization=normalization, dataset=dataset, kernel_size=kernel_size, \
                inp_channels=inp_channels, k=k, pad_conv1=pad_conv1, affine=affine)
