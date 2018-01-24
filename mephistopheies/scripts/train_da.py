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
import itertools as it

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from kaggle_camera_model_id_lib.utils import PechkaBot, ImageList, NpzFolder, NCrops, TifFolder
from kaggle_camera_model_id_lib.models import VggHead, StyleVggHead, IEEEfcn, ResNetFC, ResNetX, FatNet1
from kaggle_camera_model_id_lib.models import InceptionResNetV2fc, InceptionResNetV2fcSmall
from kaggle_camera_model_id_lib.utils import jpg_compress, equalize_v_hist, hsv_convert
from kaggle_camera_model_id_lib.utils import scale_crop_pad, gamma_correction
from kaggle_camera_model_id_lib.utils import patch_quality_dich, n_random_crops, n_pseudorandom_crops
from kaggle_camera_model_id_lib.models import DANet, ResNetFeatureExtractor, AvgFcClassifier, FCDiscriminator
from kaggle_camera_model_id_lib.models import AvgClassifier



_bot = PechkaBot()

def log(txt):
    print(txt)
    _bot.send_message(txt)
    



model_factory = {
    'resnet34_fe': lambda: ResNetFeatureExtractor(models.resnet.BasicBlock, [3, 4, 6, 3], load_resnet='resnet34'),
    'AvgFcClassifier': lambda n_classes: AvgFcClassifier(n_classes),
    'FCDiscriminator': lambda: FCDiscriminator(),
    'AvgClassifier512': lambda n_classes: AvgClassifier(n_classes, 512)
}


def process_batch_classifier(X, Y):
    optimizer_c.zero_grad()
    bs, ncrops, c, h, w = X.shape
    X = X.view(-1, c, h, w)
    Y = Y.view(ncrops*bs)
    X_var = Variable(X.cuda())
    Y_var = Variable(Y.cuda())
    log_p = model(X_var, mode='c')
    loss = criterion_c(log_p, Y_var)
    loss.backward()
    optimizer_c.step()
    loss_train_batch = loss.data[0]
    acc_train_batch = ((log_p.max(1)[1] == Y_var).float().sum()/Y_var.shape[0]).data[0]
    return loss_train_batch, acc_train_batch


def process_batch_discriminator(X_test, Y_test, X_val, Y_val):    
    optimizer_d.zero_grad()
    bs, ncrops, c, h, w = X_test.shape
    if np.prod(Y_test.shape) != ncrops*bs or np.prod(Y_val.shape) != ncrops*bs:
        return None
    X_test = X_test.view(-1, c, h, w)
    Y_test = Y_test.view(ncrops*bs)
    X_val = X_val.view(-1, c, h, w)
    Y_val = Y_val.view(ncrops*bs)    
    X = torch.cat([X_test, X_val], dim=0)
    Y = torch.cat([Y_test, Y_val], dim=0)    
    X_var = Variable(X.cuda())
    Y_var = Variable(Y.cuda())    
    p = model(X_var, mode='d')
    loss = criterion_dg(p.squeeze(), Y_var)
    loss.backward()
    optimizer_d.step()
    loss_train_batch = loss.data[0]
    acc_train_batch = (((p > 0.5).squeeze().float() == Y_var).float().sum()/Y_var.shape[0]).data[0]    
    return loss_train_batch, acc_train_batch


def process_batch_generator(X, Y):
    optimizer_g.zero_grad()
    bs, ncrops, c, h, w = X.shape
    X = X.view(-1, c, h, w)
    Y = Y.view(ncrops*bs)
    X_var = Variable(X.cuda())
    Y_var = Variable(Y.cuda())
    p = model(X_var, mode='d')
    loss = criterion_dg(p.squeeze(), Y_var)
    loss.backward()
    optimizer_d.step()
    loss_train_batch = loss.data[0]
    acc_train_batch = (((p > 0.5).squeeze().float() == Y_var).float().sum()/Y_var.shape[0]).data[0]    
    return loss_train_batch, acc_train_batch




def val_pass():
    loss_val_batch = 0
    acc_val_batch = 0
    
    for ix_batch, (X, Y) in tqdm(
        enumerate(val_loader_c), 
        total=int(len(val_loader_c.dataset.imgs)/batch_size_val_c),
        desc='Val #%i' % ix_epoch):
        
        bs, ncrops, c, h, w = X.shape
        X = X.view(-1, c, h, w)
        Y = Y.view(ncrops*bs)

        X_var = Variable(X.cuda(), volatile=True)
        Y_var = Variable(Y.cuda(), volatile=True)

        log_p = model(X_var, mode='c')
        loss = criterion_c(log_p, Y_var)

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
    test_path = cfg['test_path']
    out_dir = cfg['out_dir']
    model_path = cfg['model_path']
    crop_size = cfg['crop_size']
    step_crop_val = cfg['step_crop_val']
    n_crops_train = cfg['n_crops_train']
    batch_size_train_c = cfg['batch_size_train_c']
    batch_size_train_d = cfg['batch_size_train_d']
    batch_size_train_g = cfg['batch_size_train_g']
    batch_size_val_c = cfg['batch_size_val_c']
    workers = cfg['workers']
    n_epoches = cfg['n_epoches']
    model_type_fe = cfg['model_type_fe']
    model_type_d = cfg['model_type_d']
    model_type_c = cfg['model_type_c']
    n_classes = cfg['n_classes']    
    crop_center_size = cfg['crop_center_size']
    do_random_aug_kaggle = cfg['do_random_aug_kaggle']
    p_random_aug_kaggle_train = cfg['p_random_aug_kaggle_train']
    p_random_aug_kaggle_val = cfg['p_random_aug_kaggle_val']
    do_hard_aug = cfg['do_hard_aug']
    p_hard_aug_train = cfg['p_hard_aug_train']
    p_hard_aug_val = cfg['p_hard_aug_val']
    n_crops_search_train = cfg['n_crops_search_train']    
    learning_rate_c = cfg['learning_rate_c']
    momentum_c = cfg['momentum_c']
    weight_decay_c = cfg['weight_decay_c']
    lr_scheduler_step_size_c = cfg['lr_scheduler_step_size_c']
    lr_scheduler_gamma_c = cfg['lr_scheduler_gamma_c']
    learning_rate_g = cfg['learning_rate_g']
    momentum_g = cfg['momentum_g']
    weight_decay_g = cfg['weight_decay_g']
    lr_scheduler_step_size_g = cfg['lr_scheduler_step_size_g']
    lr_scheduler_gamma_g = cfg['lr_scheduler_gamma_g']
    learning_rate_d = cfg['learning_rate_d']
    momentum_d = cfg['momentum_d']
    weight_decay_d = cfg['weight_decay_d']
    lr_scheduler_step_size_d = cfg['lr_scheduler_step_size_d']
    lr_scheduler_gamma_d = cfg['lr_scheduler_gamma_d']
    n_iter_c = cfg['n_iter_c']
    n_iter_g = cfg['n_iter_g']
    n_iter_d = cfg['n_iter_d']
    model_path_c = cfg['model_path_c']
    n_train_iter = cfg['n_train_iter']
    n_epoches_g_fixed = cfg['n_epoches_g_fixed']
    
    
    
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
        transform_train_c = transforms.Compose([
            transforms.Lambda(lambda img: [
                aug_optional_train(aug_train(img))
                for i in range(n_crops_train)
            ]),
            transforms.Lambda(lambda crops: torch.stack([normalize(to_tensor(crop)) for crop in crops]))
        ])
    else:
        log(' -> dich fscore transform_train is selected')
        transform_train_c = transforms.Compose([
            transforms.Lambda(lambda img: [
                aug_optional_train(img) for img in aug_train_fscore(img)
            ]),
            transforms.Lambda(lambda crops: torch.stack([normalize(to_tensor(crop)) for crop in crops]))
        ])
    
    
    ds_train_c = ImageList(
        train_list_path,
        transform=transform_train_c,
        target_transform=transforms.Compose([
            transforms.Lambda(lambda y: [y]*n_crops_train),
            transforms.Lambda(lambda ylist: torch.LongTensor(ylist))
        ]))
    train_loader_c = torch.utils.data.DataLoader(    
        ds_train_c,
        batch_size=batch_size_train_c, 
        shuffle=True,
        num_workers=workers, 
        pin_memory=True)
    log('train_loader_c.size: %i' % len(train_loader_c.dataset.imgs))
    
    
    ds_train_d_test = TifFolder(
        test_path,
        transform=transforms.Compose([
            transforms.Lambda(lambda img: NCrops(np.array(img), crop_size=crop_size, step=step_crop_val)),
            transforms.Lambda(lambda crops: torch.stack([normalize(to_tensor(crop)) for crop in crops]))
        ]),
        target_transform=transforms.Compose([
            transforms.Lambda(lambda y: [0]*int(np.floor(1 + (512 - crop_size)/step_crop_val))**2),
            transforms.Lambda(lambda ylist: torch.FloatTensor(ylist))
        ]))
    train_loader_d_test = torch.utils.data.DataLoader(    
        ds_train_d_test,
        batch_size=batch_size_train_d, 
        shuffle=True,
        num_workers=workers, 
        pin_memory=False)
    log('train_loader_d_test.size: %i' % len(train_loader_d_test.dataset.imgs))
    
    
    ds_train_d_val = NpzFolder(
        val_path,
        transform=transforms.Compose([
            transforms.Lambda(lambda img: NCrops(img, crop_size=crop_size, step=step_crop_val)),
            transforms.Lambda(lambda crops: torch.stack([normalize(to_tensor(aug_optional_val(Image.fromarray(crop)))) 
                                                         for crop in crops]))
        ]),
        target_transform=transforms.Compose([
            transforms.Lambda(lambda y: [1]*int(np.floor(1 + (512 - crop_size)/step_crop_val))**2),
            transforms.Lambda(lambda ylist: torch.FloatTensor(ylist))
        ]))
    train_loader_d_val = torch.utils.data.DataLoader(    
        ds_train_d_val,
        batch_size=batch_size_train_d, 
        shuffle=True,
        num_workers=workers, 
        pin_memory=False)
    log('train_loader_d_val.size: %i' % len(train_loader_d_val.dataset.imgs))
    
    
    ds_train_g = NpzFolder(
        val_path,
        transform=transforms.Compose([
            transforms.Lambda(lambda img: NCrops(img, crop_size=crop_size, step=step_crop_val)),
            transforms.Lambda(lambda crops: torch.stack([normalize(to_tensor(aug_optional_val(Image.fromarray(crop)))) 
                                                         for crop in crops]))
        ]),
        target_transform=transforms.Compose([
            transforms.Lambda(lambda y: [0]*int(np.floor(1 + (512 - crop_size)/step_crop_val))**2),
            transforms.Lambda(lambda ylist: torch.FloatTensor(ylist))
        ]))
    train_loader_g = torch.utils.data.DataLoader(    
        ds_train_g,
        batch_size=batch_size_train_g, 
        shuffle=True,
        num_workers=workers, 
        pin_memory=False)
    log('train_loader_d_val.size: %i' % len(train_loader_g.dataset.imgs))
    
    
    ds_val_c = NpzFolder(
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
    val_loader_c = torch.utils.data.DataLoader(    
        ds_val_c,
        batch_size=batch_size_val_c,
        shuffle=False,
        num_workers=workers, 
        pin_memory=True)
    log('val_loader_c.size: %i' % len(val_loader_c.dataset.imgs))
    
    

    
    model = DANet(
        model_factory[model_type_fe](),
        model_factory[model_type_d](),
        model_factory[model_type_c](n_classes))
    if model_path is not None:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        #loss_train = checkpoint['loss_train']
        #acc_train = checkpoint['acc_train']
        #loss_val = checkpoint['loss_val']
        #acc_val = checkpoint['acc_val']
        #log('Last state:\n  TLoss: %0.6f\n  TAcc:  %0.4f\n  VLoss: %0.6f\n  VAcc:  %0.4f' % 
        #    (loss_train[-1], acc_train[-1], loss_val[-1], acc_val[-1]))
        del(checkpoint)
        log('model loaded: %s' % model_path)
    elif model_path_c is not None:
        checkpoint = torch.load(model_path_c)        
        loss_train = checkpoint['loss_train']
        acc_train = checkpoint['acc_train']
        loss_val = checkpoint['loss_val']
        acc_val = checkpoint['acc_val']
        log('Last state:\n  TLoss: %0.6f\n  TAcc:  %0.4f\n  VLoss: %0.6f\n  VAcc:  %0.4f' % 
            (loss_train[-1], acc_train[-1], loss_val[-1], acc_val[-1]))
        
        d = checkpoint['model']
        state_fe = model.feature_exctractor.state_dict()
        state_fe_update = dict([(k, v) for (k, v) in d.items() if k in state_fe])
        if len(state_fe_update) > 0:
            log(' -> feature_exctractor loaded')
            state_fe.update(state_fe_update)
            model.feature_exctractor.load_state_dict(state_fe)
            
        state_c = model.classifier.state_dict()
        state_c_update = dict([(k, v) for (k, v) in d.items() if k in state_c])
        if len(state_c_update) > 0:
            log(' -> classifier loaded')
            state_c.update(state_c_update)
            model.classifier.load_state_dict(state_c_update)
        
        del(checkpoint, d, state_fe, state_c)
        log('classifier model loaded: %s' % model_path_c)
        
    model = model.cuda()  
    
    
    
    is_g_fixed = False
    if n_epoches_g_fixed is not None:
        is_g_fixed = True
        for prm in model.feature_exctractor.parameters():
            prm.requires_grad = False
        log(' -> Generator fixed')
    
    
    criterion_c = nn.CrossEntropyLoss().cuda()
    criterion_dg = nn.BCELoss().cuda()
        
    def create_optimizer_c():
        return optim.SGD(
            filter(lambda p: p.requires_grad, 
                    it.chain(model.feature_exctractor.parameters(), model.classifier.parameters())),
            lr=learning_rate_c, 
            momentum=momentum_c, 
            dampening=0, 
            weight_decay=weight_decay_c,
            nesterov=True)
        
    optimizer_c = create_optimizer_c()
    log('optimizer_c %s:\n' % str(type(optimizer_c)) + 
        '\n'.join([('%s: %s' % (k, str(v))) for (k, v) in 
                   optimizer_c.param_groups[0].items() if k != 'params']))
    lr_scheduler_c = torch.optim.lr_scheduler.StepLR(
        optimizer_c,
        step_size=lr_scheduler_step_size_c, 
        gamma=lr_scheduler_gamma_c)

    def create_optimizer_g():
        return optim.Adam(
            filter(lambda p: p.requires_grad, model.feature_exctractor.parameters()),
            lr=learning_rate_g, 
            betas=(0.9, 0.999),
            weight_decay=weight_decay_g)
    
    if not is_g_fixed:
        optimizer_g = create_optimizer_g()
        log('optimizer_g %s:\n' % str(type(optimizer_g)) + 
            '\n'.join([('%s: %s' % (k, str(v))) for (k, v) in 
                       optimizer_g.param_groups[0].items() if k != 'params']))
        lr_scheduler_g = torch.optim.lr_scheduler.StepLR(
            optimizer_g,
            step_size=lr_scheduler_step_size_g, 
            gamma=lr_scheduler_gamma_g)
    
    optimizer_d = optim.Adam(
        model.discrimitator.parameters(), 
        lr=learning_rate_d, 
        betas=(0.9, 0.999),
        weight_decay=weight_decay_d)
    log('optimizer_d %s:\n' % str(type(optimizer_d)) + 
        '\n'.join([('%s: %s' % (k, str(v))) for (k, v) in 
                   optimizer_d.param_groups[0].items() if k != 'params']))
    lr_scheduler_d = torch.optim.lr_scheduler.StepLR(
        optimizer_d,
        step_size=lr_scheduler_step_size_d, 
        gamma=lr_scheduler_gamma_d)    
    
    
    trainin_log = []
    
    
    train_loader_c_chain = it.chain(*[train_loader_c]*n_epoches)    
    train_loader_d_test_chain = it.chain(*[train_loader_d_test]*n_epoches)
    train_loader_d_val_chain = it.chain(*[train_loader_d_val]*n_epoches)
    train_loader_g_chain = it.chain(*[train_loader_g]*n_epoches)


    for ix_epoch in range(n_epoches):
        start_time = time.time()
        lr_scheduler_c.step()
        if not is_g_fixed:
            lr_scheduler_g.step()
        lr_scheduler_d.step()
        
        if is_g_fixed and ix_epoch >= n_epoches_g_fixed:
            for prm in model.feature_exctractor.parameters():
                prm.requires_grad = True
            optimizer_c = create_optimizer_c()
            optimizer_g = create_optimizer_g()
            lr_scheduler_g = torch.optim.lr_scheduler.StepLR(
                optimizer_g,
                step_size=lr_scheduler_step_size_g, 
                gamma=lr_scheduler_gamma_g)
            is_g_fixed = False
            log(' -> Generator unfixed')
        
        model.train()
        epoch_log = defaultdict(float)
        for ix_train_iter in  tqdm(range(n_train_iter), desc='Train #%i' % ix_epoch):            
            for i in range(n_iter_d):
                X_test, Y_test = train_loader_d_test_chain.__next__()
                X_val, Y_val = train_loader_d_val_chain.__next__()
                res_tmp = process_batch_discriminator(X_test, Y_test, X_val, Y_val)
                while res_tmp is None:
                    X_test, Y_test = train_loader_d_test_chain.__next__()
                    X_val, Y_val = train_loader_d_val_chain.__next__()
                    res_tmp = process_batch_discriminator(X_test, Y_test, X_val, Y_val)
                loss_train_batch_d, acc_train_batch_d = res_tmp
                epoch_log['loss_train_d'] += loss_train_batch_d
                epoch_log['acc_train_d'] += acc_train_batch_d
            if not is_g_fixed:
                for i in range(n_iter_g):
                    X, Y = train_loader_g_chain.__next__()
                    loss_train_batch_g, acc_train_batch_g = process_batch_generator(X, Y)
                    epoch_log['loss_train_g'] += loss_train_batch_g
                    epoch_log['acc_train_g'] += acc_train_batch_g
            for i in range(n_iter_c):
                X, Y = train_loader_c_chain.__next__()
                loss_train_batch_c, acc_train_batch_c = process_batch_classifier(X, Y)
                epoch_log['loss_train_c'] += loss_train_batch_c
                epoch_log['acc_train_c'] += acc_train_batch_c
                
            if options.is_debug and ix_train_iter > 3:
                break
        if n_train_iter*n_iter_d > 0:
            epoch_log['loss_train_d'] /= n_train_iter*n_iter_d
            epoch_log['acc_train_d'] /= n_train_iter*n_iter_d
        if not is_g_fixed and n_train_iter*n_iter_g > 0:
            epoch_log['loss_train_g'] /= n_train_iter*n_iter_g
            epoch_log['acc_train_g'] /= n_train_iter*n_iter_g
        if n_train_iter*n_iter_c > 0:
            epoch_log['loss_train_c'] /= n_train_iter*n_iter_c
            epoch_log['acc_train_c'] /= n_train_iter*n_iter_c
        
        model.eval()
        loss_val, acc_val = val_pass()
        epoch_log['loss_val'] = loss_val
        epoch_log['acc_val'] = acc_val
        
        epoch_log['time'] = time.time() - start_time        
        
        trainin_log.append(dict(epoch_log.items()))
                

        log('ix_epoch: %i' % ix_epoch)
        msg = ''
        for k, v in sorted(epoch_log.items(), key=lambda t: t[0]):
            msg += '  %s: %0.6f\n' % (k, v)
        log(msg)        

        torch.save({
            'model': model.state_dict(),
            'epoch': ix_epoch + 1,
            'trainin_log': trainin_log,
            'class_to_idx': train_loader_c.dataset.class_to_idx,
            'cfg': cfg
        }, os.path.join(out_dir, 'checkpoint.tar'))

        if trainin_log[-1]['acc_val'] == np.max([d['acc_val'] for d in trainin_log]) and not is_g_fixed:
            log('Best found!')
            copyfile(
                os.path.join(out_dir, 'checkpoint.tar'),
                os.path.join(out_dir, 'best_model.tar'))

        if options.is_debug and ix_epoch == 1:
            log('END OF DEBUG')
            break