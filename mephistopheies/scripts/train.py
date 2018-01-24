import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")
plt.rcParams['figure.figsize'] = 16, 12
import pandas as pd
from tqdm import tqdm_notebook, tqdm
import io
from PIL import Image
from glob import glob
from collections import defaultdict
import os
import pickle
from optparse import OptionParser
from datetime import datetime
import json
import sys
import time
from shutil import copyfile
import cv2
cv2.ocl.setUseOpenCL(True)
import random

import imgaug as ia
from imgaug import augmenters as iaa

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from kaggle_camera_model_id_lib.utils import PechkaBot, ImageList, NpzFolder, NCrops, MultiDataset
from kaggle_camera_model_id_lib.models import VggHead, StyleVggHead, IEEEfcn, ResNetFC, ResNetX, FatNet1
from kaggle_camera_model_id_lib.models import InceptionResNetV2fc, InceptionResNetV2fcSmall
from kaggle_camera_model_id_lib.utils import jpg_compress, equalize_v_hist, hsv_convert
from kaggle_camera_model_id_lib.utils import scale_crop_pad, gamma_correction
from kaggle_camera_model_id_lib.utils import patch_quality_dich, n_random_crops, n_pseudorandom_crops



_bot = PechkaBot()

def log(txt):
    print(txt)
    _bot.send_message(txt)
    
    
def train_pass(train_loader, model, criterion, optimizer):
    loss_train_batch = 0
    acc_train_batch = 0
    
    for ix_batch, (X, Y) in tqdm(
        enumerate(train_loader), 
        total=int(len(train_loader.dataset.imgs)/batch_size_train), 
        desc='Train #%i' % ix_epoch):
        
        bs, ncrops, c, h, w = X.shape
        X = X.view(-1, c, h, w)
        Y = Y.view(ncrops*bs)
        
        X_var = Variable(X.cuda())
        Y_var = Variable(Y.cuda())

        log_p = model(X_var)
        loss = criterion(log_p, Y_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_train_batch += loss.data[0]
        acc_train_batch += ((log_p.max(1)[1] == Y_var).float().sum()/Y_var.shape[0]).data[0]
        
        if options.is_debug and ix_batch > 50:
            break
            
    X_var = X_var.cpu()
    del(X_var)
    Y_var = Y_var.cpu()
    del(Y_var)
        
    loss_train_batch /= ix_batch + 1
    acc_train_batch /= ix_batch + 1    
    
    return loss_train_batch, acc_train_batch


def val_pass(val_loader, model, criterion):
    loss_val_batch = 0
    acc_val_batch = 0
    
    for ix_batch, (X, Y) in tqdm(
        enumerate(val_loader), 
        total=int(len(val_loader.dataset.imgs)/batch_size_val), 
        desc='Val #%i' % ix_epoch):
        
        bs, ncrops, c, h, w = X.shape
        X = X.view(-1, c, h, w)
        Y = Y.view(ncrops*bs)

        X_var = Variable(X.cuda(), volatile=True)
        Y_var = Variable(Y.cuda(), volatile=True)

        log_p = model(X_var)
        loss = criterion(log_p, Y_var)

        loss_val_batch += loss.data[0]
        acc_val_batch += ((log_p.max(1)[1] == Y_var).float().sum()/Y_var.shape[0]).data[0]
        
        if options.is_debug and ix_batch > 50:
            break
            
    X_var = X_var.cpu()
    del(X_var)
    Y_var = Y_var.cpu()
    del(Y_var)
           
    loss_val_batch /= ix_batch + 1
    acc_val_batch /= ix_batch + 1
    
    return loss_val_batch, acc_val_batch



model_factory = {
    'Vgg19Head_E_2b_bn': lambda n_classes: VggHead(num_classes=n_classes, vgg_key='E_2b', load_vgg_bn=True, batch_norm=True),
    'Vgg19Head_E_3b_bn': lambda n_classes: VggHead(num_classes=n_classes, vgg_key='E_3b', load_vgg_bn=True, batch_norm=True),
    'Vgg19Head_E_bn': lambda n_classes: VggHead(num_classes=n_classes, load_vgg_bn=True, vgg_key='E', batch_norm=True),
    'Vgg11Head_A_bn': lambda n_classes: VggHead(num_classes=n_classes, load_vgg_bn=True, vgg_key='A', batch_norm=True),
    'Vgg11Head_A': lambda n_classes: VggHead(num_classes=n_classes, load_vgg_bn=True, vgg_key='A', batch_norm=False),
    'StyleVggHead_bn': lambda n_classes: StyleVggHead(num_classes=n_classes, load_vgg_bn=True),
    'IEEEfcn': lambda n_classes: IEEEfcn(n_classes),
    'resnet18fc_pretrained': lambda n_classes: ResNetFC(
        models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=n_classes, load_resnet='resnet18'),
    'resnet18fc': lambda n_classes: ResNetFC(
        models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=n_classes, load_resnet=None),
    'resnet18X_pretrained': lambda n_classes: ResNetX(
        models.resnet.BasicBlock, [2, 2, 2, 2], num_classes=n_classes, load_resnet='resnet18'),
    'InceptionResNetV2fc_5_10_4': lambda n_classes: InceptionResNetV2fc(
        num_classes=n_classes, nun_block35=5, num_block17=10, num_block8=4),
    'InceptionResNetV2fcSmall_5_10': lambda n_classes: InceptionResNetV2fcSmall(
        num_classes=n_classes, nun_block35=5, num_block17=10),
    'resnet34fc_pretrained': lambda n_classes: ResNetFC(
        models.resnet.BasicBlock, [3, 4, 6, 3], num_classes=n_classes, load_resnet='resnet34'),
    'resnet50fc_pretrained': lambda n_classes: ResNetFC(
        models.resnet.Bottleneck, [3, 4, 6, 3], num_classes=n_classes, load_resnet='resnet50'),
    'FatNet1': lambda n_classes: FatNet1(n_classes)
}

criterion_factory = {
    'CrossEntropyLoss': lambda prms: nn.CrossEntropyLoss(),
    'MultiMarginLoss': lambda prms: nn.MultiMarginLoss(**prms)
}

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-c', 
                      '--config', 
                      dest='cfg_path',
                      help='config path')
    parser.add_option('-d', 
                      '--debug', 
                      action="store_true", 
                      dest="is_debug")
    (options, args) = parser.parse_args()
    if options.cfg_path is None:
        sys.exit('cfg_path is not provided')    
    if options.is_debug:
        log('DEBUG MODE ON')
    log('-----------\n\nStarting training process: \n  %s\n  %s' % (str(datetime.now()), __file__))
    log('config: %s' % options.cfg_path)
    
    with open(options.cfg_path) as f:
        cfg = json.load(f)
        
    log('Config:')
    for k, v in cfg.items():
        log('  %s = %s' % (k, v))
    
    train_list_path = cfg['train_list_path']
    val_path = cfg['val_path']
    out_dir = cfg['out_dir']
    model_path = cfg['model_path']
    crop_size = cfg['crop_size']
    step_crop_val = cfg['step_crop_val']
    n_crops_train = cfg['n_crops_train']
    batch_size_train = cfg['batch_size_train']
    batch_size_val = cfg['batch_size_val']
    workers = cfg['workers']
    n_epoches = cfg['n_epoches']
    model_type = cfg['model_type']
    n_classes = cfg['n_classes']
    learning_rate = cfg['learning_rate']
    momentum = cfg['momentum']
    lr_scheduler_step_size = cfg['lr_scheduler_step_size']
    lr_scheduler_gamma = cfg['lr_scheduler_gamma']
    weight_decay = cfg['weight_decay']
    optim_type = cfg['optim_type']
    crop_center_size = cfg['crop_center_size']
    do_random_aug_kaggle = cfg['do_random_aug_kaggle']
    p_random_aug_kaggle_train = cfg['p_random_aug_kaggle_train']
    p_random_aug_kaggle_val = cfg['p_random_aug_kaggle_val']
    do_hard_aug = cfg['do_hard_aug']
    p_hard_aug_train = cfg['p_hard_aug_train']
    p_hard_aug_val = cfg['p_hard_aug_val']
    criterion_type = cfg['criterion_type']
    criterion_params = cfg['criterion_params']
    n_crops_search_train = cfg['n_crops_search_train']
    train_list_pseudo_npz = cfg['train_list_pseudo_npz']
    
    
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    random_crop = transforms.RandomCrop(crop_size)
    center_crop = transforms.CenterCrop(crop_center_size)
    rvf = transforms.RandomVerticalFlip()
    rhf = transforms.RandomHorizontalFlip()
    random_flip = lambda img: rvf(rhf(img))
    
    scale_05 = lambda img: scale_crop_pad(img, 0.5)
    scale_08 = lambda img: scale_crop_pad(img, 0.8)
    scale_15 = lambda img: scale_crop_pad(img, 1.5)
    scale_20 = lambda img: scale_crop_pad(img, 2.0)
    gamma_08 = lambda img: gamma_correction(img, 0.8)
    gamma_12 = lambda img: gamma_correction(img, 1.2)
    jpg_70 = lambda img: jpg_compress(img, (70, 71))
    jpg_90 = lambda img: jpg_compress(img, (90, 91))
    augs = [scale_05, scale_08, scale_15, scale_20, gamma_08, gamma_12, jpg_70, jpg_90]
    

    def random_aug_kaggle(img, p=0.5):
        if np.random.rand() < p:
            return random.choice(augs)(img)
        return img
    
    
    blur = iaa.GaussianBlur(sigma=(0, 2))
    sharpen = iaa.Sharpen(alpha=(0, 1), lightness=(0.5, 2))
    emboss = iaa.Emboss(alpha=(0, 1), strength=(0, 2))
    contrast_normalization = iaa.ContrastNormalization(alpha=(0.7, 1.3))
    hard_aug = iaa.OneOf([blur, sharpen, emboss, contrast_normalization])
    sometimes_train = iaa.Sometimes(p_hard_aug_train, hard_aug)
    sometimes_val = iaa.Sometimes(p_hard_aug_val, hard_aug)
    
    
    def aug_train(img):
        if min(img.size) > crop_center_size:
            return random_flip(random_crop(center_crop(img)))
        img_np = np.array(img)
        if img_np.shape[0] < crop_center_size and img_np.shape[1] > crop_center_size:
            n = np.random.randint(img_np.shape[1] - crop_center_size)
            return random_flip(random_crop(Image.fromarray(img_np[:, n:(n + crop_center_size), :])))
        if img_np.shape[1] < crop_center_size and img_np.shape[0] > crop_center_size:
            n = np.random.randint(img_np.shape[0] - crop_center_size)
            return random_flip(random_crop(Image.fromarray(img_np[n:(n + crop_center_size), :, :])))
        return random_flip(random_crop(img))
    
    def aug_train_fscore(img):
        if min(img.size) > crop_center_size:
            img_np = np.array(center_crop(img))
        else:
            img_np = np.array(img)
            if img_np.shape[0] < crop_center_size and img_np.shape[1] > crop_center_size:
                n = np.random.randint(img_np.shape[1] - crop_center_size)
                img_np = img_np[:, n:(n + crop_center_size), :]
            if img_np.shape[1] < crop_center_size and img_np.shape[0] > crop_center_size:
                n = np.random.randint(img_np.shape[0] - crop_center_size)
                img_np = img_np[n:(n + crop_center_size), :, :]

        crops = n_pseudorandom_crops(img_np, crop_size, n_crops_train, n_crops_search_train, patch_quality_dich)
        for img in crops:
            yield random_flip(random_crop(Image.fromarray(img)))
    
    def aug_optional_train(img):
        if do_hard_aug:
            img = Image.fromarray(sometimes_train.augment_image(np.array(img)))
        
        if do_random_aug_kaggle:
            img = random_aug_kaggle(img, p_random_aug_kaggle_train)
        return img
    
    def aug_optional_val(img):
        if do_hard_aug:
            img = Image.fromarray(sometimes_val.augment_image(np.array(img)))
        
        if do_random_aug_kaggle:
            img = random_aug_kaggle(img, p_random_aug_kaggle_val)
        return img
    
    if n_crops_search_train is None:
        log(' -> default transform_train is selected')
        transform_train = transforms.Compose([
            transforms.Lambda(lambda img: [
                aug_optional_train(aug_train(img))
                for i in range(n_crops_train)
            ]),
            transforms.Lambda(lambda crops: torch.stack([normalize(to_tensor(crop)) for crop in crops]))
        ])
    else:
        log(' -> dich fscore transform_train is selected')
        transform_train = transforms.Compose([
            transforms.Lambda(lambda img: [
                aug_optional_train(img) for img in aug_train_fscore(img)
            ]),
            transforms.Lambda(lambda crops: torch.stack([normalize(to_tensor(crop)) for crop in crops]))
        ])
    
    ds_train = ImageList(
        train_list_path,
        transform=transform_train,
        target_transform=transforms.Compose([
            transforms.Lambda(lambda y: [y]*n_crops_train),
            transforms.Lambda(lambda ylist: torch.LongTensor(ylist))
        ]))
    
    if train_list_pseudo_npz is not None:        
        ds_train_pseudo = NpzFolder(
            train_list_pseudo_npz,
            transform=transforms.Compose([
                transforms.Lambda(lambda img: [
                    aug_train(Image.fromarray(img))
                    for i in range(n_crops_train)
                ]),
                transforms.Lambda(lambda crops: torch.stack([normalize(to_tensor(crop)) for crop in crops]))
            ]),
            target_transform=transforms.Compose([
                transforms.Lambda(lambda y: [y]*n_crops_train),
                transforms.Lambda(lambda ylist: torch.LongTensor(ylist))
            ]))
        ds_train = MultiDataset([ds_train, ds_train_pseudo])
        log(' -> pseudo dataset is loaded')
    
    train_loader = torch.utils.data.DataLoader(    
        ds_train,
        batch_size=batch_size_train, 
        shuffle=True,
        num_workers=workers, 
        pin_memory=True)
    log('train_loader.size: %i' % len(train_loader.dataset.imgs))    
    
    ds_val = NpzFolder(
        val_path,
        transform=transforms.Compose([
            transforms.Lambda(lambda img: NCrops(img, crop_size=crop_size, step=step_crop_val)),
            transforms.Lambda(lambda crops: torch.stack([normalize(to_tensor(aug_optional_val(Image.fromarray(crop)))) 
                                                         for crop in crops]))
        ]),
        target_transform=transforms.Compose([
            transforms.Lambda(lambda y: [y]*int(np.floor(1 + (512 - crop_size)/step_crop_val))**2),
            transforms.Lambda(lambda ylist: torch.LongTensor(ylist))
        ]))
    val_loader = torch.utils.data.DataLoader(    
        ds_val,
        batch_size=batch_size_val, 
        shuffle=False,
        num_workers=workers, 
        pin_memory=True)
    log('val_loader.size: %i' % len(val_loader.dataset.imgs))
    
    
    model = model_factory[model_type](n_classes)
    if model_path is not None:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        loss_train = checkpoint['loss_train']
        acc_train = checkpoint['acc_train']
        loss_val = checkpoint['loss_val']
        acc_val = checkpoint['acc_val']
        log('Last state:\n  TLoss: %0.6f\n  TAcc:  %0.4f\n  VLoss: %0.6f\n  VAcc:  %0.4f' % 
            (loss_train[-1], acc_train[-1], loss_val[-1], acc_val[-1]))
        del(checkpoint)
        log('model loaded: %s' % model_path)
    model = model.cuda()
    
    
    criterion = criterion_factory[criterion_type](criterion_params)
    criterion = criterion.cuda()
    
    if optim_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
            momentum=momentum, 
            dampening=0, 
            weight_decay=weight_decay,
            nesterov=True)
    elif optim_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            betas=(0.9, 0.999),
            weight_decay=weight_decay)
        
    log('optimizer %s:\n' % str(type(optimizer)) + 
        '\n'.join([('%s: %s' % (k, str(v))) for (k, v) in 
                   optimizer.param_groups[0].items() if k != 'params']))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=lr_scheduler_step_size, 
        gamma=lr_scheduler_gamma)
    
    
    loss_train = []
    acc_train = []
    loss_val = []
    acc_val = []
    epoch_time = []


    for ix_epoch in range(n_epoches):
        start_time = time.time()
        lr_scheduler.step()

        model.train()    
        loss_train_batch, acc_train_batch = train_pass(train_loader, model, criterion, optimizer)

        model.eval()
        loss_val_batch, acc_val_batch = val_pass(val_loader, model, criterion)

        loss_train.append(loss_train_batch)
        acc_train.append(acc_train_batch)
        loss_val.append(loss_val_batch)
        acc_val.append(acc_val_batch)
        epoch_time.append(time.time() - start_time)

        log('ix_epoch: %i' % ix_epoch)
        log('  Time:  %0.2f\n  TLoss: %0.6f\n  TAcc:  %0.4f\n  VLoss: %0.6f\n  VAcc:  %0.4f' % 
            (epoch_time[-1], loss_train[-1], acc_train[-1], loss_val[-1], acc_val[-1]))

        torch.save({
            'model': model.state_dict(),
            'opt': optimizer.state_dict(),
            'epoch': ix_epoch + 1,
            'loss_train': loss_train,
            'acc_train': acc_train,
            'loss_val': loss_val,
            'acc_val': acc_val,
            'class_to_idx': train_loader.dataset.class_to_idx,
            'cfg': cfg
        }, os.path.join(out_dir, 'checkpoint.tar'))

        if acc_val[-1] == np.max(acc_val):
            log('Best found!')
            copyfile(
                os.path.join(out_dir, 'checkpoint.tar'),
                os.path.join(out_dir, 'best_model.tar'))

        if options.is_debug and ix_epoch == 1:
            log('END OF DEBUG')
            break