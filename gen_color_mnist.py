import torchvision
import torch
import numpy as np
import torchvision.transforms as transforms
import tqdm
# import matplotlib.pyplot as plt
from torch.autograd import Variable
import argparse
import os
# from PIL import Image
data_path = 'datasets/'



parser = argparse.ArgumentParser(description='Generate color MNIST')

# # Directories
# parser.add_argument('--data', type=str, default='/export/home/datasets/',
#                     help='location of the data corpus')

# Hyperparams
parser.add_argument('--cpr', nargs='+', type=float, default=[0.5,0.5],
                    help='color choices corresponding to a class with these probability')
parser.add_argument('--mode', type=str, default='fgbg',
                    help='add color to digit (fg) or background (bg)')
args = parser.parse_args()


trans = ([transforms.ToTensor()])
trans = transforms.Compose(trans)
fulltrainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=trans)
trainloader = torch.utils.data.DataLoader(fulltrainset, batch_size=2000, shuffle=False, num_workers=2, pin_memory=True)
test_set = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=trans)
testloader = torch.utils.data.DataLoader(test_set, batch_size=2000, shuffle=False, num_workers=2, pin_memory=True)
nb_classes = 10


# generate color codes
def get_color_codes(cpr):
    C = np.random.rand(len(cpr), nb_classes,3)
    # _,_,V = np.linalg.svd(C)
    # V = V[:3]
    C = C/np.max(C, axis=2)[:,:,None]
    # C *= 255
    print(C.shape)
    return C

def gen_bgcolor_data(loader, img_size=(3,28,28), cpr=[0.5, 0.5], noise=10.):
	if cpr is not None:
	    assert sum(cpr)==1, '--cpr must be a non-negative list which sums to 1'
	    C = get_color_codes(cpr)
	else:
	    C = get_color_codes([1])
	tot_iters =  len(loader)
	for i in tqdm.tqdm(range(tot_iters), total=tot_iters):
	    x, targets = next(iter(loader))
	    targets = targets.cpu().numpy()
	    bs = targets.shape[0]

	    x = (((x*255)>150)*255).type('torch.FloatTensor')
	    x_rgb = torch.ones(x.size(0),3, x.size()[2], x.size()[3]).type('torch.FloatTensor')
	    x_rgb = x_rgb* x
	    bg = (255-x_rgb)
	    # c = C[targets] if np.random.rand()>cpr else C[np.random.randint(C.shape[0], size=targets.shape[0])]
	    color_choice = np.argmax(np.random.multinomial(1, cpr, targets.shape[0]), axis=1) if cpr is not None else 0
	    c = C[color_choice,targets] if cpr is not None else C[color_choice,np.random.randint(nb_classes, size=targets.shape[0])]
	    c = c.reshape(-1, 3, 1, 1)
	    c= torch.from_numpy(c).type('torch.FloatTensor')
	    bg[:,0] = bg[:,0]* c[:,0]
	    bg[:,1] = bg[:,1]* c[:,1]
	    bg[:,2] = bg[:,2]* c[:,2]
	    x_rgb = x_rgb + bg
	    x_rgb = x_rgb + torch.tensor((noise)* np.random.randn(*x_rgb.size())).type('torch.FloatTensor')
	    x_rgb = torch.clamp(x_rgb, 0.,255.)
	    if i==0:
	        color_data_x = np.zeros((bs* tot_iters, *img_size))
	        color_data_y = np.zeros((bs* tot_iters,))
	    color_data_x[i*bs: (i+1)*bs] = x_rgb/255.
	    color_data_y[i*bs: (i+1)*bs] = targets
	return color_data_x, color_data_y


def gen_fgcolor_data(loader, img_size=(3,28,28), cpr=[0.5, 0.5], noise=10.):
	if cpr is not None:
	    assert sum(cpr)==1, '--cpr must be a non-negative list which sums to 1'
	    C = get_color_codes(cpr)
	else:
	    C = get_color_codes([1])
	tot_iters =   len(loader)
	for i in tqdm.tqdm(range(tot_iters), total=tot_iters):
	    x, targets = next(iter(loader))
	    targets = targets.cpu().numpy()
	    bs = targets.shape[0]

	    x = (((x*255)>150)*255).type('torch.FloatTensor')
	    x_rgb = torch.ones(x.size(0),3, x.size()[2], x.size()[3]).type('torch.FloatTensor')
	    x_rgb = x_rgb* x
	    # c = C[targets] if np.random.rand()>cpr else C[np.random.randint(C.shape[0], size=targets.shape[0])]
	    color_choice = np.argmax(np.random.multinomial(1, cpr, targets.shape[0]), axis=1) if cpr is not None else 0
	    c = C[color_choice,targets] if cpr is not None else C[color_choice,np.random.randint(nb_classes, size=targets.shape[0])]
	    c = c.reshape(-1, 3, 1, 1)
	    c= torch.from_numpy(c).type('torch.FloatTensor')
	    x_rgb[:,0] = x_rgb[:,0]* c[:,0]
	    x_rgb[:,1] = x_rgb[:,1]* c[:,1]
	    x_rgb[:,2] = x_rgb[:,2]* c[:,2]
	    x_rgb = x_rgb + torch.tensor((noise)* np.random.randn(*x_rgb.size())).type('torch.FloatTensor')
	    x_rgb = torch.clamp(x_rgb, 0.,255.)
	    if i==0:
	        color_data_x = np.zeros((bs* tot_iters, *img_size))
	        color_data_y = np.zeros((bs* tot_iters,))
	    color_data_x[i*bs: (i+1)*bs] = x_rgb/255.
	    color_data_y[i*bs: (i+1)*bs] = targets
	return color_data_x, color_data_y

def gen_fgbgcolor_data(loader, img_size=(3,28,28), cpr=[0.5, 0.5], noise=10.):
    if cpr is not None:
        assert sum(cpr)==1, '--cpr must be a non-negative list which sums to 1'
        Cfg = get_color_codes(cpr)
        Cbg = get_color_codes(cpr)
    else:
        Cfg = get_color_codes([1])
        Cbg = get_color_codes([1])
    tot_iters =  len(loader)
    for i in tqdm.tqdm(range(tot_iters), total=tot_iters):
        x, targets = next(iter(loader))
        targets = targets.cpu().numpy()
        bs = targets.shape[0]

        x = (((x*255)>150)*255).type('torch.FloatTensor')
        x_rgb = torch.ones(x.size(0),3, x.size()[2], x.size()[3]).type('torch.FloatTensor')
        x_rgb = x_rgb* x
        x_rgb_fg = 1.*x_rgb
        
        color_choice = np.argmax(np.random.multinomial(1, cpr, targets.shape[0]), axis=1) if cpr is not None else 0
        c = Cfg[color_choice,targets] if cpr is not None else Cfg[color_choice,np.random.randint(nb_classes, size=targets.shape[0])]
        c = c.reshape(-1, 3, 1, 1)
        c= torch.from_numpy(c).type('torch.FloatTensor')
        x_rgb_fg[:,0] = x_rgb_fg[:,0]* c[:,0]
        x_rgb_fg[:,1] = x_rgb_fg[:,1]* c[:,1]
        x_rgb_fg[:,2] = x_rgb_fg[:,2]* c[:,2]
        
        bg = (255-x_rgb)
        # c = C[targets] if np.random.rand()>cpr else C[np.random.randint(C.shape[0], size=targets.shape[0])]
        color_choice = np.argmax(np.random.multinomial(1, cpr, targets.shape[0]), axis=1) if cpr is not None else 0
        c = Cbg[color_choice,targets] if cpr is not None else Cbg[color_choice,np.random.randint(nb_classes, size=targets.shape[0])]
        c = c.reshape(-1, 3, 1, 1)
        c= torch.from_numpy(c).type('torch.FloatTensor')
        bg[:,0] = bg[:,0]* c[:,0]
        bg[:,1] = bg[:,1]* c[:,1]
        bg[:,2] = bg[:,2]* c[:,2]
        x_rgb = x_rgb_fg + bg
        x_rgb = x_rgb + torch.tensor((noise)* np.random.randn(*x_rgb.size())).type('torch.FloatTensor')
        x_rgb = torch.clamp(x_rgb, 0.,255.)
        if i==0:
            color_data_x = np.zeros((bs* tot_iters, *img_size))
            color_data_y = np.zeros((bs* tot_iters,))
        color_data_x[i*bs: (i+1)*bs] = x_rgb/255.
        color_data_y[i*bs: (i+1)*bs] = targets
    return color_data_x, color_data_y

dir_name = data_path + 'cmnist/' + args.mode + '_cmnist_cpr' + '-'.join(str(p) for p in args.cpr) + '/'
print(dir_name)
if not os.path.exists(dir_name):
	os.mkdir(dir_name)

if args.mode=='bg':
	gen_fn = gen_bgcolor_data
elif args.mode=='fg':
	gen_fn = gen_fgcolor_data
elif args.mode=='fgbg':
	gen_fn = gen_fgbgcolor_data

color_data_x, color_data_y = gen_fn(trainloader, img_size=(3,28,28), cpr=args.cpr, noise=10.)
np.save(dir_name+ '/train_x.npy', color_data_x)
np.save(dir_name+ '/train_y.npy', color_data_y)


color_data_x, color_data_y = gen_fn(testloader, img_size=(3,28,28), cpr=None, noise=10.)
np.save(dir_name + 'test_x.npy', color_data_x)
np.save(dir_name + 'test_y.npy', color_data_y)

