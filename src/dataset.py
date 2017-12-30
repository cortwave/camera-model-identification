import torch.utils.data as data
import pandas as pd
from PIL import Image
import os
import glob


def load(image):
    img = Image.open(image)
    return img


class Dataset(data.Dataset):
    def __init__(self, n_fold, n_folds, transform=None, train=True):
        valid_fold = pd.read_csv('../data/fold_{}.csv'.format(n_fold), header=None)
        if train:
            folds = list(range(n_folds))
            folds.remove(n_fold)
            train_dfs = [pd.read_csv('../data/fold_{}.csv'.format(i), header=None) for i in folds]
            df = pd.concat(train_dfs)
        else:
            df = valid_fold
        categories = sorted(os.listdir('../data/train'))
        categories_dict = { k: idx for k, idx in enumerate(categories)}
        self.images = df[0].values
        labels = df[1].values
        self.labels = [categories_dict[cat] for cat in labels]
        self.transform = transform

    def __len__(self):
        return self.images.size

    def __getitem__(self, idx):
        x = load(self.images[idx])
        if self.transform:
            x = self.transform(x)
        y = self.labels[idx]
        return x, y


class TestDataset(data.Dataset):
    def __init__(self, transform=None):
        self.images = sorted(glob.glob('../data/test/**'))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img = load(img_name)
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(img_name)
