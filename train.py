#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import yaml
import time
import math
import torchvision.transforms as transforms
import numpy as np
from skimage import io as img
from skimage import color, morphology, filters
import imageio
import os
import random
import datetime
import cv2
import dateutil.tz
import copy

import torch.nn.functional as F
from math import exp

from albumentations import HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, OneOf,\
    Compose, MultiplicativeNoise, ToSepia, ChannelDropout, ChannelShuffle, Cutout, InvertImg

import ConSinGAN.models as models
from ConSinGAN.imresize import imresize, imresize_in, imresize_to_shape

# In[3]:
torch.cuda.is_available() 

# In[4]:

'''
from PIL import __version__
print(__version__)'''


# In[5]:


from DiffAugment_pytorch import DiffAugment
# from DiffAugment_tf import DiffAugment

# In[10]:

def get_config(path):
    with open(path, 'r') as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)

# In[11]:


manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
    
dict_ = get_config('./config_256_8_64.yaml')
date_time = time.ctime()
print(date_time)


# In[12]:


def convert(list):
    return tuple(i for i in list)


# In[13]:


def np2torch_(x, nc_im=3, not_cuda=False):
    if nc_im == 3:
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255
    else:
        x = color.rgb2gray(x)
        x = x[:,:,None,None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not(not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not(not_cuda) else x.type(torch.FloatTensor)
    x = norm(x)
    return x

# In[17]:

def imresize(im,scale,opt):
    im = torch2uint8(im)
    im = imresize_in(im, scale_factor=scale)
    im = np2torch(im,opt)
    return im

# In[18]:

def imresize_to_shape(im,output_shape,opt):
    im = torch2uint8(im)
    im = imresize_in(im, output_shape=output_shape)
    im = np2torch(im,opt)
    return im

# In[19]:

def generate_samples(netG, opt, depth, noise_amp, writer, reals, iter_, n=None):
    opt.out_ = generate_dir2save(opt)
    dir2save = '{}/gen_samples_stage_{}'.format(opt.out_, depth)
    reals_shapes = [r.shape for r in reals]
    all_images = []
    try:
        os.makedirs(dir2save)
    except OSError:
        pass
    if(n is None):
    	n = opt.num_samples
    with torch.no_grad():
        for idx in range(n):
            noise = sample_random_noise(depth, reals_shapes, opt)
            sample = netG(noise, reals_shapes, noise_amp)
            all_images.append(sample)
            save_image('{}/gen_sample_{}_{}.jpg'.format(dir2save, iter_, idx), sample.detach())

        all_images = torch.cat(all_images, 0)
        all_images[0] = reals[depth].squeeze()
        grid = make_grid(all_images, nrow=min(5, n), normalize=True)
        writer.add_image('gen_images_{}'.format(depth), grid, iter_)

# In[21]:


def interpolate_samples(netG, opt, depth, noise_amp, writer, reals, iter_, fixed_noise_1, fixed_noise_2, n=10):
    opt.out_ = generate_dir2save(opt)
    dir2save = '{}/gen_samples_stage_{}'.format(opt.out_, depth)
    reals_shapes = [r.shape for r in reals]
    all_images = []
    try:
        os.makedirs(dir2save)
    except OSError:
        pass
    with torch.no_grad():
        interpol = copy.deepcopy(fixed_noise_1)
            
        sample = netG(fixed_noise_1, reals_shapes, noise_amp)
        all_images.append(sample)
        save_image('{}/int_sample_{}_{}.jpg'.format(dir2save, iter_, 0), sample.detach())
        for i in range(n-1):
            for _ in range(len(interpol)):
                interpol[_]=(fixed_noise_1[_] + fixed_noise_2[_])*((i+1.0)/n)
            sample = netG(interpol, reals_shapes, noise_amp)
            all_images.append(sample)
            save_image('{}/int_sample_{}_{}.jpg'.format(dir2save, iter_, i+1), sample.detach())
            
        sample = netG(fixed_noise_2, reals_shapes, noise_amp)
        all_images.append(sample)
        save_image('{}/int_sample_{}_{}.jpg'.format(dir2save, iter_, n+1), sample.detach())
            
        all_images = torch.cat(all_images, 0)
        all_images[0] = reals[depth].squeeze()
        grid = make_grid(all_images, nrow=min(5, n), normalize=True)
        writer.add_image('int_images_{}'.format(depth), grid, iter_)


# In[ ]:


def init_G_exp(opt):
    # generator initialization:
    netG = Generator_exp(opt).to(opt.device)
    netG.apply(models.weights_init)
    return netG
# In[30]:

def init_AC(opt, n_classes=None):
    #Auxiliary_classifier initialization:
    netAC = Auxiliary_classifier(opt, n_classes).to(opt.device)
    netAC.apply(models.weights_init)
    print(netAC)
    return netAC


# In[36]:

class Auxiliary_classifier(nn.Module):
    def __init__(self, opt, num_classes=None):
        super(Auxiliary_classifier, self).__init__()

        self.opt = opt
        N = int(opt.nfc)
        if(num_classes is None):
            self.num_classes = opt.num_imgs
        else:
            self.num_classes = num_classes
        
        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, opt)

        self.body = nn.Sequential()
        for i in range(opt.num_layer):
            block = ConvBlock(N, N, opt.ker_size, opt.padd_size, opt)
            self.body.add_module('block%d'%(i),block)

        self.tail = nn.Conv2d(N, 1, kernel_size=opt.ker_size, padding=opt.padd_size)
        self.classifier = nn.Conv2d(N, self.num_classes, kernel_size=opt.ker_size, padding=opt.padd_size)
        
    def forward(self,x):
        head = self.head(x)
        body = self.body(head)
        out_ = self.classifier(body)
        out = self.tail(body)
        return out, out_


# In[37]:


class convert_yamldict_to_object():
    def __init__(self, dict_):
        self.not_cuda = dict_['not_cuda']
        self.manualSeed = dict_['manualSeed']
        self.nfc = dict_['nfc']
        self.ker_size = dict_['ker_size']
        self.num_layer = dict_['num_layer']
        self.padd_size = dict_['padd_size']
        self.nc_im = dict_['nc_im']
        self.noise_amp = dict_['noise_amp']
        self.min_size = dict_['min_size']
        self.max_size = dict_['max_size']
        self.train_depth = dict_['train_depth']
        self.start_scale = dict_['start_scale']
        self.niter = dict_['niter']
        self.gamma = dict_['gamma']
        self.lr_g = dict_['lr_g']
        self.lr_d = dict_['lr_d']
        self.beta1 = dict_['beta1']
        self.Gsteps = dict_['Gsteps']
        self.Dsteps = dict_['Dsteps']
        self.lambda_grad = dict_['lambda_grad']
        self.lambda_vgg = dict_['lambda_vgg']
        self.alpha = dict_['alpha']
        self.activation = dict_['activation']
        self.lrelu_alpha = dict_['lrelu_alpha']
        self.batch_norm = dict_['batch_norm']
        self.naive_img = dict_['naive_img']
        self.gpu = dict_['gpu']
        self.train_mode = dict_['train_mode']
        self.lr_scale = dict_['lr_scale']
        self.train_stages = dict_['train_stages']
        self.fine_tune = dict_['fine_tune']
        self.model_dir = dict_['model_dir']
        self.curr_depth = dict_['curr_depth']
        self.w_amp = dict_['w_amp']
        self.w_noise = dict_['w_noise']
        self.train_init = dict_['train_init']
        self.num_samples = dict_['num_samples']
        self.lambda_sim = dict_['lambda_sim']
        self.temp_sim = dict_['temp_sim']
        self.lambda_clip = dict_['lambda_clip']
        self.device = dict_['device']
        self.timestamp = str(int(time.time()))
        self.noise_amp_init = dict_['noise_amp']        
        self.w_amp = dict_['w_amp']
        self.num_imgs = dict_['num_imgs']
        self.dir_name = dict_['dir_name']
        self.path = dict_['path']
        return


# In[38]:
opt = convert_yamldict_to_object(dict_)
cuda_env = torch.cuda.is_available() 
opt.device = 'cuda:0' if cuda_env else 'cpu'
opt.not_cuda = 0 if cuda_env else 1
print(opt.device, 'Not_cuda:' , opt.not_cuda)

# In[39]:


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def norm(x):
    out = (x - 0.5) * 2
    return out.clamp(-1, 1)


def convert_image_np(inp):
    if inp.shape[1]==3:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,:,:,:])
        inp = inp.numpy().transpose((1,2,0))
    else:
        inp = denorm(inp)
        inp = move_to_cpu(inp[-1,-1,:,:])
        inp = inp.numpy().transpose((0,1))

    inp = np.clip(inp,0,1)
    return inp


def generate_noise(size, num_samp=1, device='cuda',type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
        noise = upsampling(noise,size[1], size[2])
    elif type =='gaussian_mixture':
        noise1 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise2 = torch.randn(num_samp, size[0], size[1], size[2], device=device)
        noise = noise1+noise2
    elif type == 'uniform':
        noise = torch.randn(num_samp, size[0], size[1], size[2], device=device)
    else:
        raise NotImplementedError
    return noise


def upsampling(im,sx,sy):
    m = nn.Upsample(size=[round(sx),round(sy)],mode='bilinear',align_corners=True)
    return m(im)

def move_to_gpu(t):
    if (torch.cuda.is_available()):
        t = t.to(torch.device('cuda'))
    return t

def move_to_cpu(t):
    t = t.to(torch.device('cpu'))
    return t


def save_image(name, image):
    plt.imsave(name, convert_image_np(image), vmin=0, vmax=1)

def sample_random_noise(depth, reals_shapes, opt):
    noise = []
    for d in range(depth + 1):
        if d == 0:
            noise.append(generate_noise([opt.nc_im, reals_shapes[d][2], reals_shapes[d][3]],
                                         device='cuda:0').detach())
        else:
            if opt.train_mode == "generation" or opt.train_mode == "animation":
                noise.append(generate_noise([opt.nfc, reals_shapes[d][2] + opt.num_layer * 2,
                                             reals_shapes[d][3] + opt.num_layer * 2],
                                             device='cuda:0').detach())
            else:
                noise.append(generate_noise([opt.nfc, reals_shapes[d][2], reals_shapes[d][3]],
                                             device='cuda:0').detach())
    return noise


def sample_random_noise_(depth, reals_shapes, current, opt):
    noise = []
    for d in range(depth + 1):
        if d == 0:
            noise.append(generate_noise([opt.nc_im, reals_shapes[d][2], reals_shapes[d][3]],
                                         device=opt.device).detach())
        else:
            if opt.train_mode == "generation" or opt.train_mode == "animation":
                noise.append(generate_noise([opt.nfc, reals_shapes[d][2] + opt.num_layer * 2,
                                             reals_shapes[d][3] + opt.num_layer * 2],
                                             device=opt.device).detach())
            else:
                temp= generate_noise([opt.nfc, reals_shapes[d][2], reals_shapes[d][3]], num_samp= opt.num_imgs,
                                             device=opt.device).detach()
                noise.append(temp[current])
    return noise


def sample_random_noise_new(depth, reals_shapes, current, opt, num_samples=opt.num_imgs):
    noise = []
    for d in range(depth + 1):
        if d == 0:
            noise.append(generate_noise([opt.nc_im, reals_shapes[d][2], reals_shapes[d][3]], num_samp= num_samples,
                                         device=opt.device).detach())
        else:
            if opt.train_mode == "generation" or opt.train_mode == "animation":
                temp = generate_noise([opt.nfc, reals_shapes[d][2] + opt.num_layer * 2,
                                             reals_shapes[d][3] + opt.num_layer * 2],  num_samp= num_samples,
                                             device=opt.device).detach()
                noise.append(temp[current])
            else:                                
                temp= generate_noise([opt.nfc, reals_shapes[d][2], reals_shapes[d][3]], num_samp= num_samples,
                                             device=opt.device).detach()
                noise.append(temp[current])
    return noise

def sample_random_noise_new_alpha(depth, reals_shapes, current, opt, fixed_noise, num_samples=opt.num_imgs):
    noise = []
    #noise = torch.randn(num_samp, size[0], round(size[1]/scale), round(size[2]/scale), device=device)
    #noise = upsampling(fixed_noise,size[1], size[2])
    #for i in range(len(fixed_noise)):
        #print(fixed_noise[i].shape)
    for d in range(depth + 1):
        if d == 0:
            temp= generate_noise([opt.nc_im, reals_shapes[d][2], reals_shapes[d][3]], num_samp= num_samples,
                                         device=opt.device).detach()
            #print('fixed_noise[d].shape',fixed_noise[d].shape)
            #print('temp.shape',temp.shape)
            #print(fixed_noise[d][0].shape, temp[0].shape)
            for i in range(num_samples):
                temp[i] = (fixed_noise[d][0] + temp[0])*0.5
            noise.append(temp)
        else:
            if opt.train_mode == "generation" or opt.train_mode == "animation":
                temp = generate_noise([opt.nfc, reals_shapes[d][2] + opt.num_layer * 2,
                                             reals_shapes[d][3] + opt.num_layer * 2],  num_samp= num_samples,
                                             device=opt.device).detach()
                #print('fixed_noise[d].shape',fixed_noise[d].shape)
                #print('temp.shape',temp.shape)
                #print(fixed_noise[d][0].shape, temp[0].shape)
                for i in range(num_samples):
                    temp[i] = (fixed_noise[d][0] + temp[i])*0.5
                noise.append(temp)
            else:                                
                temp= generate_noise([opt.nfc, reals_shapes[d][2], reals_shapes[d][3]], num_samp= num_samples,
                                             device=opt.device).detach()
                #print(fixed_noise[d].shape) 
                for i in range(num_samples):
                    temp[i] = (fixed_noise[d][0] + temp[0])*0.5
                noise.append(temp)
    return noise

def sample_random_noise_new_(depth, reals_shapes, current, opt, num_samples=1):
    noise = []
    for d in range(depth + 1):
        if d == 0:
            noise.append(generate_noise([opt.nc_im, reals_shapes[d][2], reals_shapes[d][3]], num_samp= num_samples,
                                         device='cuda:0').detach())
        else:
            if opt.train_mode == "generation" or opt.train_mode == "animation":
                temp = generate_noise([opt.nfc, reals_shapes[d][2] + opt.num_layer * 2,
                                             reals_shapes[d][3] + opt.num_layer * 2],  num_samp= num_samples,
                                             device='cuda:0').detach()
                noise.append(temp)
            else:                                
                temp= generate_noise([opt.nfc, reals_shapes[d][2], reals_shapes[d][3]], num_samp= num_samples,
                                             device='cuda:0').detach()
                noise.append(temp)
    return noise

def sample_rand_noise(depth, reals_shapes, current, opt, num_samples=1):
    noise = []
    for d in range(depth + 1):
        if d == 0:
            noise.append(generate_noise([opt.nc_im, reals_shapes[d][2], reals_shapes[d][3]], num_samp= num_samples,
                                         device='cuda:0').detach())
    return noise

def calc_gradient_penalty_AC_(netD, real_data, fake_data, LAMBDA, device='cuda:0', num_classes=None):
    MSGGan = False
    if  MSGGan:
        alpha = torch.rand(1, 1)
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = [alpha * rd + ((1 - alpha) * fd) for rd, fd in zip(real_data, fake_data)]
        interpolates = [i.to(device) for i in interpolates]
        interpolates = [torch.autograd.Variable(i, requires_grad=True) for i in interpolates]

        disc_interpolates = netD(interpolates)
    else:
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)#.cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates, class_interpolates = netD(interpolates)
        
    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    
    gradients_ = torch.autograd.grad(outputs=class_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(class_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    #LAMBDA = 1
    gradient_penalty_disc = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    gradient_penalty_class = ((gradients_.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    gradient_penalty = (gradient_penalty_class + gradient_penalty_disc)/2.0 
    return gradient_penalty

def calc_gradient_penalty_AC_backup(netD, real_data, fake_data, LAMBDA, device='cuda:0'):
    MSGGan = False
    if  MSGGan:
        alpha = torch.rand(1, 1)
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = [alpha * rd + ((1 - alpha) * fd) for rd, fd in zip(real_data, fake_data)]
        interpolates = [i.to(device) for i in interpolates]
        interpolates = [torch.autograd.Variable(i, requires_grad=True) for i in interpolates]

        disc_interpolates = netD(interpolates)
    else:
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)#.cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates, class_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    
    gradients_ = torch.autograd.grad(outputs=class_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(class_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    #LAMBDA = 1
    gradient_penalty_disc = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    gradient_penalty_class = ((gradients_.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    gradient_penalty = (gradient_penalty_class + gradient_penalty_disc)/2.0 
    return gradient_penalty

def calc_gradient_penalty_AC(netD, real_data, fake_data, LAMBDA, device='cuda:0'):
    MSGGan = False
    if  MSGGan:
        alpha = torch.rand(1, 1)
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = [alpha * rd + ((1 - alpha) * fd) for rd, fd in zip(real_data, fake_data)]
        interpolates = [i.to(device) for i in interpolates]
        interpolates = [torch.autograd.Variable(i, requires_grad=True) for i in interpolates]

        disc_interpolates = netD(interpolates)
    else:
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)#.cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates, _ = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),#.cuda(), #if use_cuda else torch.ones(
                                  #disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def read_image(opt, input_name):
    x = img.imread('%s' % (input_name))
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]
    return x

def read_image_dir(dir, opt):
    x = img.imread(dir)
    x = np2torch(x,opt)
    x = x[:,0:3,:,:]
    return x


def np2torch(x, opt):
    if opt.nc_im == 3:
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255
    else:
        x = color.rgb2gray(x)
        x = x[:,:,None,None]
        x = x.transpose(3, 2, 0, 1)
    x = torch.from_numpy(x)
    if not(opt.not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    x = norm(x)
    return x


def torch2uint8(x):
    x = x[0,:,:,:]
    x = x.permute((1,2,0))
    x = 255*denorm(x)
    x = x.cpu().numpy()
    x = x.astype(np.uint8)
    return x

def save_networks(netG, netDs, z, opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    if isinstance(netDs, list):
        for i, netD in enumerate(netDs):
            torch.save(netD.state_dict(), '%s/netD_%s.pth' % (opt.outf, str(i)))
    else:
        torch.save(netDs.state_dict(), '%s/netD.pth' % (opt.outf))
    try:
        torch.save(z, '%s/z_opt.pth' % (opt.outf))
    except:
        pass

def adjust_scales2image(real_, opt):
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3]]),1)
    real = imresize(real_, opt.scale1, opt)

    opt.stop_scale = opt.train_stages - 1
    opt.scale_factor = math.pow(opt.min_size / (min(real.shape[2], real.shape[3])), 1 / opt.stop_scale)
    return real


def create_reals_pyramid(real, opt):
    reals = []
    # use old rescaling method for harmonization
    if opt.train_mode == "harmonization":
        for i in range(opt.stop_scale):
            scale = math.pow(opt.scale_factor, opt.stop_scale - i)
            curr_real = imresize(real, scale, opt)
            reals.append(curr_real)
    # use new rescaling method for all other tasks
    else:
        for i in range(opt.stop_scale):
            scale = math.pow(opt.scale_factor,((opt.stop_scale-1)/math.log(opt.stop_scale))*math.log(opt.stop_scale-i)+1)
            curr_real = imresize(real,scale,opt)
            reals.append(curr_real)
    reals.append(real)
    return reals


# In[40]:


def load_trained_model(opt):
    dir = generate_dir2save(opt)

    if os.path.exists(dir):
        Gs = torch.load('%s/Gs.pth' % dir, map_location="cuda:{}".format(torch.cuda.current_device()))
        Zs = torch.load('%s/Zs.pth' % dir, map_location="cuda:{}".format(torch.cuda.current_device()))
        reals = torch.load('%s/reals.pth' % dir, map_location="cuda:{}".format(torch.cuda.current_device()))
        NoiseAmp = torch.load('%s/NoiseAmp.pth' % dir, map_location="cuda:{}".format(torch.cuda.current_device()))
    else:
        print('no trained model exists: {}'.format(dir))

    return Gs,Zs,reals,NoiseAmp

# In[41]:


def generate_dir2save(opt):
    training_image_name = opt.dir_name
    dir2save = './TrainedModels/{}/'.format(training_image_name)
    dir2save += opt.timestamp
    dir2save += "_{}".format(opt.train_mode)
    if opt.train_mode == "harmonization" or opt.train_mode == "editing":
        if opt.fine_tune:
            dir2save += "_{}".format("fine-tune")
    dir2save += "_train_depth_{}_lr_scale_{}".format(opt.train_depth, opt.lr_scale)
    if opt.batch_norm:
        dir2save += "_BN"
    dir2save += "_act_" + opt.activation
    if opt.activation == "lrelu":
        dir2save += "_" + str(opt.lrelu_alpha)

    return dir2save


# In[42]:


def post_config(opt):
    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:{}".format(opt.gpu))
    opt.noise_amp_init = opt.noise_amp
    opt.timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt


# In[43]:
def load_config(opt):
    if not os.path.exists(opt.model_dir):
        print("Model not found: {}".format(opt.model_dir))
        exit()

    with open(os.path.join(opt.model_dir, 'parameters.txt'), 'r') as f:
        params = f.readlines()
        for param in params:
            param = param.split("-")
            param = [p.strip() for p in param]
            param_name = param[0]
            param_value = param[1]
            try:
                param_value = int(param_value)
            except ValueError:
                try:
                    param_value = float(param_value)
                except ValueError:
                    pass
            setattr(opt, param_name, param_value)
    return opt

# In[47]:


def generate_gif(dir2save, netG, fixed_noise, reals, noise_amp, opt, alpha=0.1, beta=0.9, start_scale=1,
                 num_images=100, fps=10):
    def denorm_for_gif(img):
        img = denorm(img).detach()
        img = img[0, :, :, :].cpu().numpy()
        img = img.transpose(1, 2, 0) * 255
        img = img.astype(np.uint8)
        return img

    reals_shapes = [r.shape for r in reals]
    all_images = []

    with torch.no_grad():
        noise_random = sample_random_noise(len(fixed_noise) - 1, reals_shapes, opt)
        z_prev1 = [0.99 * fixed_noise[i] + 0.01 * noise_random[i] for i in range(len(fixed_noise))]
        z_prev2 = fixed_noise
        for _ in range(num_images):
            noise_random = sample_random_noise(len(fixed_noise)-1, reals_shapes, opt)
            diff_curr = [beta*(z_prev1[i]-z_prev2[i])+(1-beta)*noise_random[i] for i in range(len(fixed_noise))]
            z_curr = [alpha * fixed_noise[i] + (1 - alpha) * (z_prev1[i] + diff_curr[i]) for i in range(len(fixed_noise))]

            if start_scale > 0:
                z_curr = [fixed_noise[i] for i in range(start_scale)] + [z_curr[i] for i in range(start_scale, len(fixed_noise))]

            z_prev2 = z_prev1
            z_prev1 = z_curr

            sample = netG(z_curr, reals_shapes, noise_amp)
            sample = denorm_for_gif(sample)
            all_images.append(sample)
    imageio.mimsave('{}/start_scale={}_alpha={}_beta={}.gif'.format(dir2save, start_scale, alpha, beta), all_images, fps=fps)

# In[50]:
trans_hflip = transforms.RandomHorizontalFlip(p=1.0)
trans_vflip = transforms.RandomVerticalFlip(p=1.0)
def get_real(_reals, depth, current, add_noise=True, num_samp=1, w_noise=0.01, h_flip=False, v_flip=False):
    if(add_noise):
        noise = torch.randn_like(_reals[current][depth], device='cuda')
        real = _reals[current][depth] + (noise*w_noise)
    else:
        real = _reals[current][depth]
    if(isinstance(real, torch.FloatTensor) or isinstance(real, torch.cuda.FloatTensor)):
        if(h_flip and v_flip):
            real = trans_hflip(real)
            real = trans_vflip(real)
        elif(h_flip):
            real = trans_hflip(real)
        elif(v_flip):
            real = trans_vflip(real)
    return real


# In[51]:

def get_noise_amp(noise_amps, depth, current):
    if(depth==0):
        return 1
    return noise_amps[current]

# In[52]:


def update_noise_amp(noise_amps, rec_loss, depth, current, w_amp=0.9):
    if(depth==0):
        return
    n_amp = noise_amps[current]
    RMSE = torch.sqrt(rec_loss).detach() # RMSE or Root of MSE
    noise_amps[current][-1] = w_amp * n_amp[-1] + (1.0-w_amp)* 0.5 * RMSE
    return


# In[53]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_activation(opt):
    activations = {"lrelu": nn.LeakyReLU(opt.lrelu_alpha, inplace=True),
                   "elu": nn.ELU(alpha=1.0, inplace=True),
                   "prelu": nn.PReLU(num_parameters=1, init=0.25),
                   "selu": nn.SELU(inplace=True)
                   }
    return activations[opt.activation]


def upsample(x, size):
    x_up =  torch.nn.functional.interpolate(x, size=size, mode='bicubic', align_corners=True)
    return x_up


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, opt, generator=False):
        super(ConvBlock,self).__init__()
        self.add_module('conv', nn.Conv2d(in_channel, out_channel, kernel_size=ker_size, stride=1, padding=padd))
        if generator and opt.batch_norm:
            self.add_module('norm', nn.BatchNorm2d(out_channel))
        self.add_module(opt.activation, get_activation(opt))

# In[55]:


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride = 1, padding=0, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# In[60]:


class Generator_exp(nn.Module):
    def __init__(self, opt):
        super(Generator_exp, self).__init__()

        self.opt = opt
        N = int(opt.nfc)

        self._pad = nn.ZeroPad2d(1)
        self._pad_block = nn.ZeroPad2d(opt.num_layer-1) if opt.train_mode == "generation"\
                                                           or opt.train_mode == "animation" \
                                                        else nn.ZeroPad2d(opt.num_layer)

        self.head = ConvBlock(opt.nc_im, N, opt.ker_size, opt.padd_size, opt, generator=True)

        self.body = torch.nn.ModuleList([])
        _first_stage = nn.Sequential()
        for i in range(opt.num_layer):
            block = ConvBlock(N, N, opt.ker_size, opt.padd_size, opt, generator=True)
            _first_stage.add_module('block%d'%(i),block)
        self.body.append(_first_stage)

        self.tail = nn.Sequential(
            nn.Conv2d(N, opt.nc_im, kernel_size=opt.ker_size, padding=opt.padd_size),
            nn.Tanh())

    def init_next_stage(self):
        self.body.append(copy.deepcopy(self.body[-1]))

    def forward(self, noise, real_shapes, noise_amp):
        x = self.head(self._pad(noise[0]))

        # we do some upsampling for training models for unconditional generation to increase
        # the image diversity at the edges of generated images
        if self.opt.train_mode == "generation" or self.opt.train_mode == "animation":
            x = upsample(x, size=[x.shape[2] + 2, x.shape[3] + 2])
        x = self._pad_block(x)
        x_prev_out = self.body[0](x)

        for idx, block in enumerate(self.body[1:], 1):
            if self.opt.train_mode == "generation" or self.opt.train_mode == "animation":
                x_prev_out_1 = upsample(x_prev_out, size=[real_shapes[idx][2], real_shapes[idx][3]])
                x_prev_out_2 = upsample(x_prev_out, size=[real_shapes[idx][2] + self.opt.num_layer*2,
                                                          real_shapes[idx][3] + self.opt.num_layer*2])
                x_prev = block(x_prev_out_2)
            else:
                x_prev_out_1 = upsample(x_prev_out, size=real_shapes[idx][2:])
                x_prev = block(self._pad_block(x_prev_out_1))
            x_prev_out = x_prev + x_prev_out_1

        out = self.tail(self._pad(x_prev_out))
        return out

# In[61]:

def train_single_scale_l2_AC_Gen_eff_aug(netAC, netG, reals, fixed_noise, noise_amp, opt, depth, writer=None):
    policy = 'color'
    _reals = reals
    current = np.random.randint(len(_reals))
    reals = _reals[current]
    reals_shapes = [real.shape for real in reals]
    print('reals_shapes:', reals_shapes)
    real = reals[depth]
    alpha = opt.alpha

    ce_loss = nn.CrossEntropyLoss()
    ############################
    # define z_opt for training on reconstruction
    ###########################
    _fixed_noise = []
    print('length of fixed noise', len(fixed_noise))
    criterion_l1 = nn.L1Loss()
    criterion_l2 = nn.MSELoss()
    
    for iter_ in range(len(_reals)):
        reals= _reals[iter_%len(_reals)]
        real = reals[depth]
        if depth == 0:
            temp=[]
            if opt.train_mode == "generation" or opt.train_mode == "retarget":
                z_opt = reals[0]
            elif opt.train_mode == "animation":
                z_opt = generate_noise([opt.nc_im, reals_shapes[depth][2], reals_shapes[depth][3]],
                                                 device=opt.device).detach()
            temp.append(z_opt.detach())
            _fixed_noise.append(temp)
        else:
            _fixed_noise = fixed_noise
    
    if(depth!=0):
        print('length of _fixed noise', len(_fixed_noise))
        
    ############################
    # define optimizers, learning rate schedulers, and learning rates for lower stages
    ###########################
    # setup optimizers for D
    optimizerAC = optim.Adam(netAC.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))

    # setup optimizers for G
    # remove gradients from stages that are not trained
    for block in netG.body[:-opt.train_depth]:
        for param in block.parameters():
            param.requires_grad = False

    # set different learning rate for lower stages
    parameter_list = [{"params": block.parameters(), "lr": opt.lr_g * (opt.lr_scale**(len(netG.body[-opt.train_depth:])-1-idx))}
               for idx, block in enumerate(netG.body[-opt.train_depth:])]

    # add parameters of head and tail to training
    if depth - opt.train_depth < 0:
        parameter_list += [{"params": netG.head.parameters(), "lr": opt.lr_g * (opt.lr_scale**depth)}]
    parameter_list += [{"params": netG.tail.parameters(), "lr": opt.lr_g}]
    optimizerG = optim.Adam(parameter_list, lr=opt.lr_g, betas=(opt.beta1, 0.999))

    # define learning rate schedules
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[0.8*opt.niter], gamma=opt.gamma)
    schedulerAC = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerAC, milestones=[0.8*opt.niter], gamma=opt.gamma)

    ############################
    # calculate noise_amp
    ###########################
    #noise_amps
    _z_reconstruction = []
    if depth == 0:
        _noise_amp = []
        for _ in range(len(_reals)):
            noise_amp = []
            noise_amp.append(torch.tensor(1.,device=opt.device).unsqueeze_(0))
            _noise_amp.append(noise_amp)
    else:
        _noise_amp = noise_amp
        
    print('noise_amp', noise_amp)
    # start training
    _iter = tqdm(range(opt.niter))
    
    n_samples = len(_reals)
    #n_samples = 2
    print('Samples size:',n_samples)
    
    for iter in _iter:
        _iter.set_description('stage [{}/{}]:'.format(depth, opt.stop_scale))
        temp = []
        classes_ = []
        permute = np.random.permutation(np.arange(len(_reals)))
        for curr in range(n_samples):
            idx = permute[curr]
            reals = _reals[idx%len(_reals)]
            real = reals[depth]
            temp.append(real)
            classes_.append(idx%len(_reals))
        
        classes = torch.tensor(torch.from_numpy(np.array(classes_))).to(opt.device)
        real_orig = torch.cat(temp, 0)
        # prepare data        
        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################

        for j in range(opt.Dsteps):
            current = np.random.randint(len(_reals))
            fixed_noise = _fixed_noise[current]
            noise_amp = _noise_amp[current]
            permute = np.random.permutation(np.arange(n_samples))
            real_ = copy.deepcopy(real_orig.clone().detach())
            
            for i in range(n_samples):
                real_[i] = real_orig[permute[i]]
            
            ############################
            # (0) sample noise for unconditional generation
            ############################
            noise = sample_random_noise_new_(depth, reals_shapes, current, opt, num_samples=n_samples)
            # train with real
            netAC.zero_grad()
            real_aug = DiffAugment(real_, policy=policy)
            output_real, pred_classes_real = netAC(real_)
            errAC_real = -output_real.mean()
            errAC_real_ = torch.mean(pred_classes_real, [2,3])
            class_loss_real = ce_loss(errAC_real_, classes)

            # train with fake
            if j == opt.Dsteps - 1:
                fake = netG(noise, reals_shapes, noise_amp)
            else:
                with torch.no_grad():
                    fake = netG(noise, reals_shapes, noise_amp)

            fake_aug = DiffAugment(fake, policy=policy)
            output_fake, pred_classes_fake = netAC(fake.detach())
            errAC_fake = output_fake.mean()
            errAC_fake_ = torch.mean(pred_classes_fake, [2,3])
            uniform = torch.randperm(n_samples).to(opt.device)
            uniform = uniform%len(_reals)
            class_loss_fake = ce_loss(errAC_fake_, uniform)
            
            gradient_penalty = calc_gradient_penalty_AC_(netAC, real_aug, fake_aug, opt.lambda_grad, opt.device, len(_reals))
            errAC_total = errAC_real + errAC_fake + class_loss_real + gradient_penalty + class_loss_fake 
            errAC_total.backward()
            optimizerAC.step()
            
        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################
        fake_aug = DiffAugment(fake, policy=policy)
        output, pred_classes = netAC(fake)
        errAC_classes = torch.mean(pred_classes, [2,3])
        class_loss = ce_loss(errAC_classes, uniform)
        errG = -output.mean()
        
        rec_loss = 0
        class_loss = 0
        if alpha != 0:
            for _ in range(len(_reals)):
                noise_amp = _noise_amp[_]
                rec = netG(_fixed_noise[_], reals_shapes, noise_amp)
                output_rec, pred_rec = netAC(rec)
                err_rec = output_rec.mean()
                reals = _reals[_%len(_reals)]
                real = reals[depth]
                rec_loss += criterion_l2(rec, real)
        
        netG.zero_grad()
        errG_total = errG + alpha*rec_loss
        errG_total.backward()

        for _ in range(opt.Gsteps):
            optimizerG.step()

        ############################
        # (3) Log Results
        ###########################
        try:
            if iter % 250 == 0 or iter+1 == opt.niter:
                writer.add_scalar('Loss/train/D/real/{}'.format(j), -errAC_real.item(), iter+1)
                writer.add_scalar('Loss/train/D/fake/{}'.format(j), errAC_fake.item(), iter+1)
                writer.add_scalar('Loss/train/D/gradient_penalty/{}'.format(j), gradient_penalty.item(), iter+1)
                writer.add_scalar('Loss/train/G/gen', errG.item(), iter+1)
                writer.add_scalar('Loss/train/G/reconstruction', rec_loss.item(), iter+1)
        except:
            pass
        if iter % 500 == 0 or iter+1 == opt.niter:
            print('Saving images')
            save_image('{}/fake_sample_{}.jpg'.format(opt.outf, iter+1), fake.detach())
            save_image('{}/reconstruction_{}.jpg'.format(opt.outf, iter+1), rec.detach())
            generate_samples(netG, opt, depth, noise_amp, writer, reals, iter+1)
            rand_1 = np.random.randint(len(_reals))
            rand_2 = np.random.randint(len(_reals))
            interpolate_samples(netG, opt, depth, noise_amp, writer, reals, iter+1, _fixed_noise[rand_1], _fixed_noise[rand_2])
            print('Saved images')

        schedulerAC.step()
        schedulerG.step()
        # break

    save_networks(netG, netAC, None, opt)
    return _fixed_noise, _noise_amp, netG, netAC


# In[124]:
path=opt.path
num_imgs=opt.num_imgs
if(len(path)>0):
    reals = []
    temp = np.ndarray([opt.max_size,opt.max_size,3])
    files = os.listdir(path)
    for f in files[:num_imgs]:
        real = read_image(opt, path+f)
        real = imresize_to_shape(real, temp.shape, opt)
        real = adjust_scales2image(real, opt)
        reals_ = create_reals_pyramid(real, opt)
        reals.append(reals_,)
    reals = convert(reals)
    #opt.num_imgs=len(files)

# In[ ]:

# uncomment the various input names to select different images
print("Training model with the following parameters:")
print("\t number of stages: {}".format(opt.train_stages))
print("\t number of concurrently trained stages: {}".format(opt.train_depth))
print("\t learning rate scaling: {}".format(opt.lr_scale))
print("\t non-linearity: {}".format(opt.activation))
print('Training started at ',time.ctime())

# In[128]:

start_time = time.ctime()
generator = init_G_exp(opt)
fixed_noise = []
noise_amp = []
net_D = []
net_C = []


# In[129]:
resume_scale = 0
for scale_num in range(resume_scale, opt.stop_scale+1):
    opt.out_ = generate_dir2save(opt)
    opt.outf = '%s/%d' % (opt.out_,scale_num)
    print(opt.outf)
    
    try:
        os.makedirs(opt.outf)
    except OSError:
        print(OSError)
    
    d_curr = init_AC(opt,n_classes=opt.num_imgs)
    
    if scale_num > 0:
        generator.init_next_stage()
    writer = SummaryWriter(log_dir=opt.outf)    
    with open(opt.out_+'/config_256_8_64.yaml', 'w') as f:
        for item in dict_:
            line = str(item)+': '+str(dict_.get(item))+'\n'
            f.write(line)
    print('Last saved at ',time.ctime())
    
    if(opt.num_imgs>2):
        opt.niter = int(500*(opt.num_imgs)*((opt.train_stages+opt.num_imgs)/2+(scale_num-1)))
    else:
        opt.niter = int(1000*((opt.train_stages+opt.num_imgs)/2+(scale_num-1)))
    opt.niter = 10
        
    fixed_noise, noise_amp, generator, d_curr = train_single_scale_l2_AC_Gen_eff_aug(d_curr, generator, reals, fixed_noise, noise_amp, opt, scale_num, writer)    
    try:
        torch.save(generator, '%s/G.pth' % (opt.out_))
    except:
        pass
    try:
        torch.save(fixed_noise, '%s/fixed_noise.pth' % (opt.out_))
    except:
        pass
    
    try:
        torch.save(noise_amp, '%s/noise_amp.pth' % (opt.out_))
    except:
        pass
    #net_D.append(d_curr)
    #del d_curr
    #del c_curr
    
writer.close()
end_time = time.ctime()
print('Training started at',start_time, 'and completed at ',end_time)


# In[ ]:

with open(opt.out_+'/config.yaml', 'w') as f:
    for item in dict_:
        line = str(item)+': '+str(dict_.get(item))+'\n'
        f.write(line)
print('Last saved at ',time.ctime())

# In[ ]:
#
# change the parameters in config.yaml file
#
# To test multiple images uncomment the lines
