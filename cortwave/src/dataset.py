import torch.utils.data as data
import pandas as pd
from skimage.io import imread
import cv2
import os
import glob
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed


def load(image):
    try:
        img = imread(image)
    except Exception:
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape == (2,):
        return img[0]
    else:
        return img


def load_cached(idx, img, limit):
    if idx < limit:
        return load(img)
    else:
        return img


class Dataset(data.Dataset):
    def __init__(self, n_fold, cached_part=0.5, transform=None, train=True):
        if train:
            n_folds = len(glob.glob('../data/fold_*.csv'))
            folds = list(range(n_folds))
            folds.remove(n_fold)
            train_dfs = [pd.read_csv('../data/fold_{}.csv'.format(i), header=None) for i in folds]
            df = pd.concat(train_dfs)
            self.size = len(df) * 25
        else:
            valid_fold = pd.read_csv('../data/fold_{}.csv'.format(n_fold), header=None)
            df = valid_fold
            self.size = len(df) * 25
        self.cached_limit = int(len(df) * cached_part)
        categories = sorted(os.listdir('../data/train'))
        categories_dict = {k: idx for idx, k in enumerate(categories)}
        images_names = df[0].values
        self.images_names = images_names
        self.images = Parallel(n_jobs=8)(
            delayed(load_cached)(idx, x, self.cached_limit) for idx, x in tqdm(enumerate(images_names),
                                                                               total=len(images_names),
                                                                               desc='images loading'))
        labels = df[1].values
        self.labels = [categories_dict[cat] for cat in labels]
        self.transform = transform
        self.num_classes = len(categories)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        idx = idx % len(self.images)
        x = self.images[idx] if idx < self.cached_limit else load(self.images[idx])
        if self.transform:
            manip = 'manip' in self.images_names[idx]
            x = self.transform(x, manip)
        y = self.labels[idx]
        return x, y


class TestDataset(data.Dataset):
    def __init__(self, transform=None):
        self.images = sorted(glob.glob('../data/test/**'))
        self.transform = transform
        categories = sorted(os.listdir('../data/train'))
        categories_dict = {idx: k for idx, k in enumerate(categories)}
        self.num_classes = len(categories)
        self.inverse_dict = categories_dict

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = load(img_name)
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(img_name)


class InternValidDataset(data.Dataset):
    def __init__(self, transform=None):
        df = pd.read_csv('../../validation/external_validation.csv'.format(n_fold), header=None)
        self.cached_limit = len(df)
        categories = sorted(os.listdir('../data/train'))
        categories_dict = {k: idx for idx, k in enumerate(categories)}
        images_names = df[0].values
        images_names = np.array(list(map(lambda x: '../../validation/{}'.format(x), images_names)))
        self.images_names = images_names
        self.images = Parallel(n_jobs=8)(
            delayed(load_cached)(idx, x, self.cached_limit) for idx, x in tqdm(enumerate(images_names),
                                                                               total=len(images_names),
                                                                               desc='images loading'))
        labels = df[1].values
        self.labels = [categories_dict[cat] for cat in labels]
        self.transform = transform
        self.num_classes = len(categories)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx] if idx < self.cached_limit else load(self.images[idx])
        if self.transform:
            manip = 'manip' in self.images_names[idx]
            x = self.transform(x, manip)
        y = self.labels[idx]
        return x, y
