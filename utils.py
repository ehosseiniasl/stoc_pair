
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

import torch
from torch.optim import Optimizer

import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.autograd as autograd
# import cPickle as pickle
import numpy
import torchvision.transforms as transforms
import torchvision
from torch._six import container_abcs, string_classes, int_classes
import tqdm
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def idx2onehot(idx, n, h=1, w=1):

    assert torch.max(idx).item() < n
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)

    onehot = torch.zeros(idx.size(0), n).cuda()
    onehot.scatter_(1, idx, 1)
    if h*w>1:
        onehot = onehot.view(idx.size(0), n, 1, 1)
        onehot_tensor = torch.ones(idx.size(0), n, h, w).cuda()
        onehot = onehot_tensor* onehot
    return onehot



def get_exchanged_idx(labels):
    label_idx_dict = {}
    for i in range(labels.shape[0]):
        if labels[i] in label_idx_dict:
            label_idx_dict[labels[i]].append(i)
        else:
            label_idx_dict.update({labels[i]: [i]})
    # print(label_idx_dict)
    target_idx = np.zeros((labels.shape[0],))
    for i in range(labels.shape[0]):
        cand = label_idx_dict[labels[i]]
        if len(cand)==1:
            target_idx[i] = i
        else:
            done = False
            while not done:
                idx = np.random.randint(len(cand))
                if cand[idx]!=i:
                    done = True
                    target_idx[i] =cand[idx]
    return target_idx

def stoc_recontr_loss(epoch, img1_list, img2_list, labels=None, spool=0,cpool=False, within_class=True):
    loss = 0
    for i in range(len(img1_list)):
        if epoch>i*40:
            img1 = img1_list[i]
            img2 = img2_list[i]
            if cpool:
                img1 = torch.mean(torch.abs(img1), dim=1)
                img2 = torch.mean(torch.abs(img2), dim=1)
            if spool>0:
                pool_ = nn.MaxPool2d((spool), stride=1, padding = spool//2)
                img1 = pool_(torch.abs(img1))
                img2 = pool_(torch.abs(img2))
            if i==0:
                if within_class:
                    exchanged_label_idx = get_exchanged_idx(labels)
                else:
                    exchanged_label_idx = np.random.permutation(img1.shape[0])
            # print(img2.size())
            stoc_target_img = img2[exchanged_label_idx]
            loss += torch.sum( (stoc_target_img-img1)**2 )/img1.size()[0]
    return loss



def _split_train_val(trainset, val_fraction=0, nsamples=-1):
    if nsamples>-1:
        n_train, n_val = int(nsamples), len(trainset)-int(nsamples) 
    else:    
        #n_train, n_val = int((1. - val_fraction) * len(trainset)), int(val_fraction * len(trainset))
        n_train = int((1. - val_fraction) * len(trainset))
        n_val = len(trainset) - n_train
    train_subset, val_subset = torch.utils.data.random_split(trainset, (n_train, n_val))
    return train_subset, val_subset


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        # print(img.size())
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class add_gaussian_noise():
    def __init__(self, std):
        self.std = std
    def __call__(self,x):
        noise = self.std*torch.randn_like(x)
        return x + noise



class get_dataset():
    def __init__(self, root=None, train=True):
        self.root = root
        self.train = train
        self.transform = transforms.ToTensor()
        if self.train:
            train_data_path = os.path.join(root, 'train_data')
            train_labels_path = os.path.join(root, 'train_labels')
            self.train_data = numpy.load(open(train_data_path, 'rb')) # , encoding='bytes'
            self.train_data = torch.from_numpy(self.train_data.astype('float32'))
            self.train_labels = numpy.load(open(train_labels_path, 'rb')).astype('int')
        else:
            test_data_path = os.path.join(root, 'test_data')
            test_labels_path = os.path.join(root, 'test_labels')
            self.test_data = numpy.load(open(test_data_path, 'rb'))
            self.test_data = torch.from_numpy(self.test_data.astype('float32'))
            self.test_labels = numpy.load(open(test_labels_path, 'rb')).astype('int')

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]


        return img, target




_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 86.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

"""Dataset setting and data loader for MNIST-M.
Modified from
https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
"""


import errno
import os

import torch
import torch.utils.data as data
from PIL import Image


class MNISTM(data.Dataset):
    """`MNIST-M Dataset."""

    url = "https://github.com/VanushVaswani/keras_mnistm/releases/download/1.0/keras_mnistm.pkl.gz"

    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'mnist_m_train.pt'
    test_file = 'mnist_m_test.pt'

    def __init__(self,
                 root, mnist_root="data",
                 train=True,
                 transform=None, target_transform=None,
                 download=False):
        """Init MNIST-M dataset."""
        super(MNISTM, self).__init__()
        self.root = os.path.expanduser(root)
        self.mnist_root = os.path.expanduser(mnist_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = \
                torch.load(os.path.join(self.root,
                                        self.processed_folder,
                                        self.training_file))
        else:
            self.test_data, self.test_labels = \
                torch.load(os.path.join(self.root,
                                        self.processed_folder,
                                        self.test_file))

    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.squeeze().numpy(), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Return size of dataset."""
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root,
                                           self.processed_folder,
                                           self.training_file)) and \
            os.path.exists(os.path.join(self.root,
                                        self.processed_folder,
                                        self.test_file))

    def download(self):
        """Download the MNIST data."""
        # import essential packages
        from six.moves import urllib
        import gzip
        import pickle
        from torchvision import datasets

        # check if dataset already exists
        if self._check_exists():
            return

        # make data dirs
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # download pkl files
        print('Downloading ' + self.url)
        filename = self.url.rpartition('/')[2]
        file_path = os.path.join(self.root, self.raw_folder, filename)
        if not os.path.exists(file_path.replace('.gz', '')):
            data = urllib.request.urlopen(self.url)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        # load MNIST-M images from pkl file
        with open(file_path.replace('.gz', ''), "rb") as f:
            mnist_m_data = pickle.load(f, encoding='bytes')
        mnist_m_train_data = torch.ByteTensor(mnist_m_data[b'train'])
        mnist_m_test_data = torch.ByteTensor(mnist_m_data[b'test'])

        # get MNIST labels
        mnist_train_labels = datasets.MNIST(root=self.mnist_root,
                                            train=True,
                                            download=True).train_labels
        mnist_test_labels = datasets.MNIST(root=self.mnist_root,
                                           train=False,
                                           download=True).test_labels

        # save MNIST-M dataset
        training_set = (mnist_m_train_data, mnist_train_labels)
        test_set = (mnist_m_test_data, mnist_test_labels)
        with open(os.path.join(self.root,
                               self.processed_folder,
                               self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root,
                               self.processed_folder,
                               self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')
