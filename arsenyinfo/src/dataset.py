from glob import glob
from os.path import join
from collections import defaultdict
from functools import reduce, partial

from keras.utils import to_categorical
from cv2 import imread
import numpy as np
import pandas as pd

from src.aug import crop, center_crop
from src.utils import logger


class Dataset:
    def __init__(self, n_fold, batch_size, size=384, transform=None, train=True, aug=None, center_crop_size=1024,
                 fast=False):
        self.transform = transform
        self.batch_size = batch_size
        self.augment = aug
        self.size = size
        self.center_crop_size = center_crop_size
        self.fast = fast

        data, self.classes, self.class_index = self.get_data()
        val_data = data.pop(n_fold)
        if train:
            self.data = reduce(lambda x, y: x + y, data.values())
        else:
            self.data = val_data
        self.data = np.array(self.data)

    @staticmethod
    def imread(x):
        try:
            img = imread(x)
        except Exception:
            logger.exception(f'Can not read {x}')
            return
        if img is not None:
            return img
        return

    @staticmethod
    def get_data():
        raise NotADirectoryError('Implement real get_data method first.')

    def __next__(self):
        x_data, y_data = [], []
        while len(x_data) < self.batch_size:
            idx = np.random.randint(0, self.data.shape[0])
            x, y = self.data[idx]
            need_safe_aug = 'manip' in x
            x = self.imread(x)

            if x is None:
                continue

            if self.center_crop_size:
                x = center_crop(x, self.center_crop_size)

            if self.fast:
                # two crops from one image just to feed a network a bit faster
                x1 = crop(x, self.size * 2)
                x2 = crop(x, self.size * 2)

                if self.augment is not None:
                    x1, x2 = map(partial(self.augment, safe=need_safe_aug), (x1, x2))

                x_data.append(x1)
                x_data.append(x2)
                y_data.append(y)
                y_data.append(y)

            else:
                x = crop(x, self.size * 2)
                if self.augment is not None:
                    x = self.augment(x)

                x_data.append(x)
                y_data.append(y)

        x_data, y_data = np.array(x_data).astype('float32'), np.array(y_data).astype('uint8')
        if self.transform:
            x_data = self.transform(x_data)
        y_data = to_categorical(y_data, len(self.classes))
        return x_data, y_data


class KaggleDataset(Dataset):
    @staticmethod
    def get_data():
        categories = sorted(glob('/media/ssd/data/train/*'))
        cat_names = [x.split('/')[-1] for x in categories]
        cat_index = {k: i for i, k in enumerate(cat_names)}

        acc = defaultdict(list)
        i = 0
        for c in categories:
            files = glob(c + '/*')
            y_idx = cat_index[c.split('/')[-1]]

            for f in files:
                fold = i % 5
                i += 1
                acc[fold].append((f, y_idx))

        return acc, cat_names, cat_index


class PseudoDataset(Dataset):
    @staticmethod
    def get_data():
        categories = sorted(glob('/media/ssd/data/train/*'))
        cat_names = [x.split('/')[-1] for x in categories]
        cat_index = {k: i for i, k in enumerate(cat_names)}

        acc = defaultdict(list)
        i = 0
        for c in categories:
            files = glob(c + '/*')
            y_idx = cat_index[c.split('/')[-1]]

            for f in files:
                fold = i % 5
                i += 1
                acc[fold].append((f, y_idx))

        df = pd.read_csv('result/probas.csv')
        df = df[df.fname.str.contains('unalt')]
        threshold = df[cat_names].values.max() * .999
        fname = df.pop('fname')
        labels = [(f, row.argmax()) for f, (i, row) in zip(fname, df.iterrows())
                  if row.max() > threshold]
        labels = sorted(labels, key=lambda x: x[1])

        for i, (k, v) in enumerate(labels):
            fold = i % 5
            f = join('data/test/', k)
            y_idx = cat_index[v]
            acc[fold].append((f, y_idx))

        return acc, cat_names, cat_index


class PseudoOnlyDataset(Dataset):
    @staticmethod
    def get_data():
        categories = sorted(glob('/media/ssd/data/train/*'))
        cat_names = [x.split('/')[-1] for x in categories]
        cat_index = {k: i for i, k in enumerate(cat_names)}

        acc = defaultdict(list)

        df = pd.read_csv('result/probas.csv')
        df = df[df.fname.str.contains('unalt')]
        threshold = df[cat_names].values.max() * .999
        fname = df.pop('fname')
        labels = [(f, row.argmax()) for f, (i, row) in zip(fname, df.iterrows())
                  if row.max() > threshold]
        labels = sorted(labels, key=lambda x: x[1])

        for i, (k, v) in enumerate(labels):
            fold = i % 5
            f = join('data/test/', k)
            y_idx = cat_index[v]
            acc[fold].append((f, y_idx))

        return acc, cat_names, cat_index


class DresdenDataset(Dataset):
    @staticmethod
    def parse_class(s):
        return s.split('/')[-1][::-1].split('_', 2)[-1][::-1]

    def get_data(self):
        files = glob('data/dresden/*JPG')
        cat_names = sorted(list(set([self.parse_class(x) for x in files])))
        cat_index = {k: i for i, k in enumerate(cat_names)}

        logger.info(f'DresdenDataset classes: {cat_names}; {len(files)} samples')

        acc = defaultdict(list)
        i = 0

        for f in files:
            y = self.parse_class(f)
            y_idx = cat_index[y]

            fold = i % 5
            i += 1
            acc[fold].append((f, y_idx))

        return acc, cat_names, cat_index


class VisionDataset(Dataset):
    @staticmethod
    def list_files():
        return glob('data/vision/*/images/flat/*.jpg') + glob('data/vision/*/images/nat/*.jpg')

    @staticmethod
    def parse_class(s):
        return s.split('/')[2]

    def get_data(self):
        files = self.list_files()
        cat_names = sorted(list(set([self.parse_class(x) for x in files])))
        cat_index = {k: i for i, k in enumerate(cat_names)}

        logger.info(f'VisionDataset classes: {cat_names}; {len(files)} samples')

        acc = defaultdict(list)
        i = 0

        for f in files:
            y = self.parse_class(f)
            y_idx = cat_index[y]

            fold = i % 5
            i += 1
            acc[fold].append((f, y_idx))

        return acc, cat_names, cat_index


class MixedDataset(Dataset):
    @staticmethod
    def list_files():
        return glob('data/vision/*/images/flat/*.jpg') \
               + glob('data/vision/*/images/nat/*.jpg') \
               + glob('data/dresden/*JPG')

    @staticmethod
    def parse_class(s):
        if 'vision' in s:
            return s.split('/')[2]
        return s.split('/')[-1][::-1].split('_', 2)[-1][::-1]

    def get_data(self):
        files = self.list_files()
        cat_names = sorted(list(set([self.parse_class(x) for x in files])))
        cat_index = {k: i for i, k in enumerate(cat_names)}

        logger.info(f'MixedDataset classes: {cat_names}; {len(files)} samples')

        acc = defaultdict(list)
        i = 0

        for f in files:
            y = self.parse_class(f)
            y_idx = cat_index[y]

            fold = i % 5
            i += 1
            acc[fold].append((f, y_idx))

        return acc, cat_names, cat_index


def get_dataset(dataset):
    datasets = {'kaggle': (KaggleDataset, 10),
                'pseudo': (PseudoDataset, 10),
                'pseudo_only': (PseudoOnlyDataset, 10),
                'vision': (VisionDataset, 24),
                'dresden': (DresdenDataset, 27),
                'mixed': (MixedDataset, 51)}
    return datasets[dataset]


if __name__ == '__main__':
    dataset = PseudoDataset(n_fold=0,
                            transform=None,
                            batch_size=16,
                            train=False,
                            size=384,
                            aug=None,
                            center_crop_size=1024)
    x, y = next(dataset)
    print(x.shape, y.shape)
