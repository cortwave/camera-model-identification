#!/usr/bin/python3

import glob
import os
from tqdm import tqdm

FOLDS_COUNT = 5

if __name__ == '__main__':
    categories = sorted(os.listdir('../../data/train'))
    folds = [([], []) for _ in range(FOLDS_COUNT)]
    for category in tqdm(categories):
        images = glob.glob('../../data/train/{}/**'.format(category))
        images = sorted([os.path.abspath(x) for x in images])
        for idx, img in enumerate(images):
            fold = idx % FOLDS_COUNT
            folds[fold][0].append(img)
            folds[fold][1].append(category)
    for idx, fold in tqdm(enumerate(folds)):
        with open('../../data/fold_{}.csv'.format(idx), 'w') as f:
            for img, category in zip(fold[0], fold[1]):
                f.write('{},{}\n'.format(img, category))
