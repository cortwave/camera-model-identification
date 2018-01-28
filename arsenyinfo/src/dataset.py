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

VOTING = 'data/voting.csv'


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
        raise NotImplementedError('Implement real get_data method first.')

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


class MixedDataset(Dataset):
    def get_data(self):
        # main data
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

        logger.info(f'{i} samples come from the main train dataset')

        # voting based pseudo labels
        df = pd.read_csv(VOTING)
        df = df[df.votes > 6].sort_values('best_model').reset_index()[['fname', 'best_model']]

        for i, row in df.iterrows():
            fold = i % 5
            k = row['fname']
            v = row['best_model']
            f = join('data/test/', k)
            y_idx = cat_index[v]
            acc[fold].append((f, y_idx))
        logger.info(f'{i} samples come from the pseudo labels dataset')

        # Vision dataset
        files = self.list_vision_files()

        i = 0
        for f in files:
            y = self.parse_vision_class(f)
            if y is None:
                continue
            y_idx = cat_index[y]
            fold = i % 5
            i += 1
            acc[fold].append((f, y_idx))
        logger.info(f'{i} samples come from the vision dataset')

        return acc, cat_names, cat_index

    @staticmethod
    def list_vision_files():
        # return glob('data/vision/*/images/flat/*.jpg') + glob('data/vision/*/images/nat/*.jpg')
        return glob('data/vision/*/images/nat/*.jpg')

    @staticmethod
    def parse_vision_class(s):
        cls = s.split('/')[2][:3]
        if cls in {'D02', 'D10'}:
            return 'iPhone-4s'
        if cls in {'D06', 'D15'}:
            return 'iPhone-6'
        return


class PseudoOnlyDataset(Dataset):
    @staticmethod
    def get_data():
        categories = sorted(glob('/media/ssd/data/train/*'))
        cat_names = [x.split('/')[-1] for x in categories]
        cat_index = {k: i for i, k in enumerate(cat_names)}

        acc = defaultdict(list)

        df = pd.read_csv(BEST_PROBAS)
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


class SiameseWrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __next__(self):
        x_data, y_data = next(self.dataset)
        batch_size = self.dataset.batch_size // 2

        _, w, h, c = x_data.shape
        x1_batch, x2_batch = np.empty(shape=(batch_size, w, h, c), dtype=x_data.dtype), \
                             np.empty(shape=(batch_size, w, h, c), dtype=x_data.dtype)
        y_batch = np.zeros(shape=(batch_size, 1))

        for i in np.arange(batch_size):
            idx = np.arange(batch_size)
            np.random.shuffle(idx)

            a, b = 2 * i, 2 * i + 1
            x1_batch[i, :, :, :] = x_data[a]
            x2_batch[i, :, :, :] = x_data[b]
            y_batch[i] = 0 if np.all(np.equal(y_data[a], y_data[b])) else 1

        return [x1_batch, x2_batch], y_batch


def get_dataset(dataset):
    datasets = {'kaggle': (KaggleDataset, 10),
                'pseudo_only': (PseudoOnlyDataset, 10),
                'vision': (VisionDataset, 24),
                'dresden': (DresdenDataset, 27),
                'mixed': (MixedDataset, 10)}
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
