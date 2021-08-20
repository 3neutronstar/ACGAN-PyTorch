import torch
import numpy as np
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# compute the current classification accuracy
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc

def denormalize(image_tensor, dataset):
    channel_num = 0 
    if dataset == 'cifar10':
        # mean = np.array([0.4914, 0.4822, 0.4465])
        mean = np.array([0.5, 0.5, 0.5])
        # std = np.array([0.2023, 0.1994, 0.2010])
        std = np.array([0.5, 0.5, 0.5])
        channel_num = 3 
    elif dataset == 'imagenet':
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        channel_num = 3 

    for c in range(channel_num):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c]*s+m, 0, 1)

    return image_tensor