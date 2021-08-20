"""
Code modified from PyTorch DCGAN examples: https://github.com/pytorch/examples/tree/master/dcgan
"""
from __future__ import print_function
import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from utils import denormalize, weights_init, compute_acc
from network import _netG, _netD, _netD_CIFAR10, _netG_CIFAR10
from folder import ImageFolder
import tqdm
import torchvision.utils as vutils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | imagenet')
    parser.add_argument('--nz', type=int, default=110, help='size of the latent z vector')
    parser.add_argument('--eval_epoch', type=int, default=None)
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--n_images', type=int, default=1, help='number of images you want to generate')
    parser.add_argument('--outf', default='./training_data', help='folder to output images and model checkpoints')
    parser.add_argument('--gpu_id', type=int, default=0, help='The ID of the specified GPU')
    parser.add_argument('--manualSeed', type=int,default=0, help='manual seed')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes for AC-GAN')

    opt = parser.parse_args()

    # specify the gpu id if using only 1 gpu
    if opt.ngpu == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)
    cudnn.benchmark = True
    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # some hyper parameters
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    num_classes = int(opt.num_classes)
    # Define the generator and initialize the weights
    if opt.dataset == 'imagenet':
        netG = _netG(ngpu, nz)
    else:
        netG = _netG_CIFAR10(ngpu, nz)
    if opt.dataset == 'imagenet':
        netD = _netD(ngpu, num_classes)
    else:
        netD = _netD_CIFAR10(ngpu, num_classes)
    
    try:
        netG_state_dict=torch.load(os.path.join(opt.outf,f'netG_epoch_{opt.eval_epoch}.pth'))
        netD_state_dict=torch.load(os.path.join(opt.outf,f'netD_epoch_{opt.eval_epoch}.pth'))
        netG.load_state_dict(netG_state_dict)
        netD.load_state_dict(netD_state_dict)
    except:
        raise NotImplementedError
    noise = torch.FloatTensor(1, nz, 1, 1)
    aux_label = torch.LongTensor(1)
    if opt.cuda:
        netG.cuda()
        netD.cuda()
        noise,aux_label=noise.cuda(),aux_label.cuda()

    num_generated_images=[0 for _ in range(num_classes)]
    i=0
    if not os.path.exists(os.path.join(opt.outf,'gen_images')):
        os.mkdir(os.path.join(opt.outf,'gen_images'))
    for cls_gen in range(num_classes):
        if not os.path.exists(os.path.join(opt.outf,'gen_images',f'c{cls_gen}')):
            os.mkdir(os.path.join(opt.outf,'gen_images',f'c{cls_gen}'))

    while sum(num_generated_images)<opt.n_images:
        cls_gen=i%num_classes # which class you want to generate
        if num_generated_images[cls_gen]<=(opt.n_images//num_classes):
            class_onehot = np.zeros(num_classes)
            class_onehot[cls_gen]=1
            noise_ = np.random.normal(0, 1, (1, nz))
            noise_[0,:num_classes] = class_onehot
            noise_ = (torch.from_numpy(noise_))
            noise.data.copy_(noise_.view(1, nz, 1, 1))
            fake = netG(noise)

            if torch.argmax(netD(fake)[1])==cls_gen:
                print(f'\r [{sum(num_generated_images)}/{opt.n_images}] saving images complete',end='')
                #save image
                vutils.save_image(denormalize(fake,opt.dataset).squeeze(0),os.path.join(opt.outf,'gen_images',f'c{cls_gen}',f'{i}.png'))
                num_generated_images[cls_gen]+=1
            else:
                print(f'fail to save class {cls_gen} when i is {i}')
        i+=1
        

if __name__=='__main__':
    main()