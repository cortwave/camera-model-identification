import torch.utils.data as data
import pandas as pd
import cv2
import os
import glob
from tqdm import tqdm
import numpy as np
from joblib import Parallel, delayed


def crop_center(img, crop=512):
    if img.shape[0] > crop or img.shape[1] > crop:
        y, x = img.shape[:2]
        startx = x // 2 - (crop // 2)
        starty = y // 2 - (crop // 2)
        return img[starty:starty + crop, startx:startx + crop, :]
    else:
        return img


def load(image, crop_central=False):
    img = cv2.imread(image)
    if img.shape == (2,):
        img = img[0]
    if img.shape[0] < img.shape[1]:
        img = np.rot90(img).copy()
    return crop_center(img) if crop_central else img


def load_cached(idx, img, limit, crop_central):
    if idx < limit:
        return load(img, crop_central)
    else:
        return img


class Dataset(data.Dataset):
    def __init__(self, n_fold, cached_part=0.0, transform=None, train=True, crop_central=False):
        self.crop_central = crop_central
        if train:
            n_folds = len(glob.glob('../data/fold_*.csv'))
            folds = list(range(n_folds))
            folds.remove(n_fold)
            train_dfs = [pd.read_csv('../data/fold_{}.csv'.format(i), header=None) for i in folds]
            df = pd.concat(train_dfs)
            self.size = len(df) * 5
        else:
            valid_fold = pd.read_csv('../data/fold_{}.csv'.format(n_fold), header=None)
            df = valid_fold
            self.size = len(df) * 5
        self.cached_limit = int(len(df) * cached_part)
        categories = sorted(os.listdir('../data/train'))
        categories_dict = {k: idx for idx, k in enumerate(categories)}
        images_names = df[0].values
        self.images_names = images_names
        self.images = Parallel(n_jobs=8)(
            delayed(load_cached)(idx, x, self.cached_limit, crop_central) for idx, x in tqdm(enumerate(images_names),
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
        x = self.images[idx] if idx < self.cached_limit else load(self.images[idx], self.crop_central)
        y = self.labels[idx]
        if self.transform:
            manip = 'manip' in self.images_names[idx]
            x = self.transform(x, manip, y)
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
            manip = 'manip' in img_name
            img = self.transform(img, manip)
        return img, os.path.basename(img_name)


class InternValidDataset(data.Dataset):
    def __init__(self, transform=None):
        df = pd.read_csv('../../validation/external_validation.csv')
        self.cached_limit = len(df)
        categories = sorted(os.listdir('../data/train'))
        categories_dict = {k: idx for idx, k in enumerate(categories)}
        images_names = df['fname'].values
        images_names = np.array(list(map(lambda x: '../../validation/{}'.format(x), images_names)))
        self.images_names = images_names
        self.images = Parallel(n_jobs=8)(
            delayed(load_cached)(idx, x, self.cached_limit) for idx, x in tqdm(enumerate(images_names),
                                                                               total=len(images_names),
                                                                               desc='images loading'))
        labels = df['camera'].values
        self.labels = [categories_dict[cat] for cat in labels]
        self.transform = transform
        self.num_classes = len(categories)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.images[idx] if idx < self.cached_limit else load(self.images[idx])
        if self.transform:
            x = self.transform(x)
        y = self.labels[idx]
        return x, y
