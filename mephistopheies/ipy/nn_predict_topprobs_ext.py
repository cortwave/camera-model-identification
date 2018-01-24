from glob import glob
import os
from shutil import copyfile
import time
import numpy as np
import math
from tqdm import tqdm
from datetime import datetime
import sys
from optparse import OptionParser
import json
import h5py
from collections import defaultdict
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, Bottleneck
from torch.utils import model_zoo
import torchvision.datasets as datasets
import torch.optim.lr_scheduler

from kaggle_cdiscount_icc_lib.utils import get_mean_image, ImageFolderExFiles, Cat2TreeIndex, find_classes
from kaggle_cdiscount_icc_lib.utils import NormalizeMeanImage, C2Tree3hTargetTransform
from kaggle_cdiscount_icc_lib.models import ResNet50_3h_inc
from kaggle_cdiscount_icc_lib.utils import PechkaBot, ImageList

from se_resnet import se_resnet50


_bot = PechkaBot()

def log(txt):
    print(txt)
    _bot.send_message(txt)

def create_resnet50():
    model = models.resnet50(pretrained=False)    
    model.avgpool = nn.AvgPool2d(avgpool_size)
    model.fc = nn.Linear(
        model.fc.in_features, 
        n_classes)
    return model

def create_ResNet50_3h_inc():
    model = ResNet50_3h_inc(
        num_classes_1=49, 
        num_classes_2=486, 
        num_classes_3=n_classes, 
        avgpool_size=avgpool_size, 
        pretrained=False)
    return model

def create_seresnet50():
    return se_resnet50(n_classes)

model_factory = {
    'resnet50': create_resnet50,
    'ResNet50_3h_inc': create_ResNet50_3h_inc,
    'seresnet50': create_seresnet50
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
    parser.add_option('-o', 
                      '--holdout', 
                      action="store_true", 
                      dest="skip_holdout")
    parser.add_option('-t', 
                      '--test', 
                      action="store_true", 
                      dest="skip_test")
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
    
    holdout_dir_path = cfg['holdout_dir_path']

    out_dir = cfg['out_dir']
    model_path = cfg['model_path']
    img_size = cfg['img_size']
    batch_size = cfg['batch_size']
    workers = cfg['workers']
    avgpool_size = cfg['avgpool_size_map'][str(img_size)]
    model_type = cfg['model_type']
    n_p_max = cfg['n_p_max']
    cum_p_max = cfg['cum_p_max']
    p_min = cfg['p_min']

    
    h5_holdout_path = os.path.join(out_dir, 'holdout.hdf5')
    log('h5_holdout_path: %s' % h5_holdout_path)


    img_mean = get_mean_image(norm=False, size=img_size)
    

    toTensor = transforms.ToTensor()
    normalizeMeanImage = NormalizeMeanImage(img_mean)
    
    
    holdout_loader = torch.utils.data.DataLoader(
        ImageFolderExFiles(
            holdout_dir_path, 
            transforms.Compose([
                transforms.CenterCrop(img_size),
                NormalizeMeanImage(img_mean),
                transforms.ToTensor()      
            ])),
        batch_size=batch_size, 
        shuffle=False,
        num_workers=workers, 
        pin_memory=True)
    holdout_idx2class = dict([(v, int(k)) for (k, v) in holdout_loader.dataset.class_to_idx.items()])
    log('holdout_idx2class.len: %i' % len(holdout_idx2class))
    log('holdout_loader.size: %i' % len(holdout_loader.dataset.imgs))    

    
    checkpoint = torch.load(model_path)
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        for k, v in sorted(metrics.items(), key=lambda t: t[0]):
            log('%s: %0.6f' % (k, v[-1] if len(v) > 0 else np.nan))
    else:
        loss_train = checkpoint['loss_train']
        acc_train = checkpoint['acc_train']
        loss_val = checkpoint['loss_val']
        acc_val = checkpoint['acc_val']
        log('Last state:\n  TLoss: %0.6f\n  TAcc:  %0.4f\n  VLoss: %0.6f\n  VAcc:  %0.4f' % 
            (loss_train[-1], acc_train[-1], loss_val[-1], acc_val[-1]))    
    
    c_cti = None
    if 'class_to_idx' in checkpoint:
        c_cti = 'class_to_idx'
    elif 'train.class_to_idx' in checkpoint:
        c_cti = 'train.class_to_idx'
    if 'c2ti' in checkpoint:
        c2ti = checkpoint['c2ti']
        idx2class = c2ti.l3_cats
        par2 = [c2ti.l1_cats_inv[c2ti.c2tree[c2ti.l3_cats[k3]]['category_level1']] 
                for k3 in c2ti.l3_cats.keys()]
        par3 = [c2ti.l2_cats_inv[c2ti.c2tree[c2ti.l3_cats[k3]]['category_level2']] 
                for k3 in c2ti.l3_cats.keys()]
    else:
        idx2class = dict([(v, k) for (k, v) in checkpoint[c_cti].items()])
    n_classes = len(idx2class)
    log('model.idx2class.len: %i' % n_classes)
    
    model = model_factory[model_type]()
    log('model created: %s' % model_type)
    
    
    log('loading model: %s' % model_path)
    checkpoint = torch.load(model_path)

    pretrained_dict = checkpoint['model']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)    

    
    log('model loaded')
    del(checkpoint, pretrained_dict, model_dict)
    model.cuda()
    model.eval()
    
    softmax = nn.Softmax(dim=1)
    
    if options.skip_holdout:
        log('HOLDOUT SKIPPED')
    else:
        log('starting holdout prediction...')        
        dt_h5_str = h5py.special_dtype(vlen=str)
        h5_holdout = h5py.File(h5_holdout_path, 'w')
        acc_holdout_batch = 0
        for ix_batch, (X, Y, files) in tqdm(
            enumerate(holdout_loader), 
            total=int(len(holdout_loader.dataset.imgs)/batch_size), 
            desc='Holdout'):

            y_true = np.array(list(map(lambda i: holdout_idx2class[i], Y))).astype(np.int32)

            g = h5_holdout.create_group(str(ix_batch))
            files = [os.path.basename(s).encode() for s in files]
            g.create_dataset(
                'files', 
                (len(files),), 
                dtype=dt_h5_str, 
                data=files)

            g.create_dataset(
                't', 
                y_true.shape, 
                dtype='i', 
                data=y_true)



            X_var = Variable(X.cuda(), volatile=True)

            log_p = model(X_var)

            if isinstance(log_p, list):
                #log_p = log_p[-1]
                log_p_c1, log_p_c2, log_p_c3 = log_p
                log_p = F.log_softmax(log_p_c3) + \
                        F.log_softmax(log_p_c2)[:, par3] + \
                        F.log_softmax(log_p_c1)[:, par2]


            p = softmax(log_p).cpu().data.numpy()

            df = pd.DataFrame(np.vstack(p).astype(np.float32))
            df['file_id'] = np.array(range(y_true.shape[0])).astype(np.int32)
            df = pd.melt(df, id_vars=['file_id'], var_name='class_id', value_name='p')
            df['class_id'] = df['class_id'].astype(np.int32)
            df = df.loc[df['p'] > p_min]
            df.sort_values(['file_id', 'p'], ascending=[True, False], inplace=True)
            df = df.groupby(['file_id', ]).head(n_p_max)
            df['p_cum'] = df.groupby(['file_id'])['p'].agg(np.cumsum)
            df = df.set_index(['file_id']).join(
                pd.DataFrame(df.groupby(['file_id'])['p'].max()), 
                rsuffix='_max').reset_index()
            df = df[((df['p_cum'] < cum_p_max) | (df['p'] == df['p_max']))].drop(['p_cum'], axis=1)
            df['class_id'] = df['class_id'].apply(lambda k: idx2class[k]).astype(np.int32)

            y_pred = df.groupby(['file_id']).head(1)['class_id']

            g.create_dataset(
                'y', 
                y_pred.shape, 
                dtype='i', 
                data=y_pred)

            g.create_dataset(
                'f_id', 
                df['file_id'].shape, 
                dtype='i', 
                data=df['file_id'])
            g.create_dataset(
                'c_id', 
                df['class_id'].shape, 
                dtype='i', 
                data=df['class_id'])
            g.create_dataset(
                'p', 
                df['p'].shape, 
                dtype='f', 
                data=df['p'])

            acc_holdout_batch += (y_true == y_pred).sum()/y_true.shape[0]

            if options.is_debug and ix_batch == 9:
                log('END OF DEBUG')
                break


        acc_holdout_batch /= ix_batch + 1
        h5_holdout.attrs['n_batches'] = ix_batch + 1
        h5_holdout.attrs['acc'] = acc_holdout_batch
        h5_holdout.close()
        log('holdout acc: %0.6f' % acc_holdout_batch)
    
    

    log('done!')