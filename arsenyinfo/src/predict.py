from glob import glob
from functools import reduce, partial
from os import path

import pandas as pd
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras.models import load_model
from fire import Fire
from tqdm import tqdm
import cv2
import numpy as np
from glog import logger

from src.aug import jpg_compress, gamma_correction


def _crop(img, shape, option):
    margin = 512 - shape
    half = int(margin / 2)
    crops = [lambda x: x[:-margin, :-margin, ...],
             lambda x: x[:-margin, margin:, ...],
             lambda x: x[margin:, margin:, ...],
             lambda x: x[margin:, :-margin, ...],
             lambda x: x[half:-half, half:-half, ...],
             ]

    return crops[option](img)


def get_test_files(shape):
    files = glob('data/test/*')
    for f in files:
        img = cv2.imread(f)
        name = f.split('/')[-1]
        safe = 'manip' in name
        crops = batch_aug([_crop(img, shape, x) for x in range(5)], safe=safe)
        crops = preprocess_input(crops.astype('float32'))
        yield name, crops


def batch_aug(imgs, safe=True):
    a, b, c = imgs[0].shape

    functions = [lambda x: x, np.flipud, np.fliplr, partial(np.rot90, k=1), partial(np.rot90, k=3)]

    if not safe:
        functions += [jpg_compress, gamma_correction]

    batch = np.empty((len(functions) * len(imgs), a, b, c), dtype=np.uint8)
    for i, f in enumerate(functions):
        for j, img in enumerate(imgs):
            idx = i * len(imgs) + j
            batch[idx] = f(img)
    return batch


def describe_model(m):
    logger.info(f'File {m} created: {path.getctime(m)}')
    logger.info(f'File {m} modified: {path.getmtime(m)}')
    logger.info(f'File {m} accessed: {path.getatime(m)}')
    return m


def predict(model_name, shape=384):
    classes = ['HTC-1-M7', 'LG-Nexus-5x', 'Motorola-Droid-Maxx', 'Motorola-Nexus-6', 'Motorola-X',
               'Samsung-Galaxy-Note3', 'Samsung-Galaxy-S4', 'Sony-NEX-7', 'iPhone-4s', 'iPhone-6']

    load_model_ = partial(load_model, custom_objects={'relu6': relu6,
                                                      'DepthwiseConv2D': DepthwiseConv2D})

    models = glob(f'result/models/{model_name}*.h5')
    result = []
    models = [load_model_(model, compile=False) for model in map(describe_model, models)]

    probas = []

    for name, batch in tqdm(get_test_files(shape=shape), desc='predictions processed', total=len(glob('data/test/*'))):
        pred = np.array(reduce(lambda x, y: x + y, [model.predict(batch) for model in models])).sum(axis=0)
        idx = pred.argmax()
        result.append({'fname': name, 'camera': classes[idx]})

        proba = {k: v for k, v in zip(classes, pred)}
        proba['fname'] = name
        probas.append(proba)

    df = pd.DataFrame(result)

    unalt = df.fname.apply(lambda x: 'unalt' in x)
    manip = df.fname.apply(lambda x: 'manip' in x)

    unalt_df = df.copy()
    unalt_df['camera'][manip] = 'tmp'

    manip_df = df.copy()
    manip_df['camera'][unalt] = 'tmp'

    unalt_df.to_csv('result/submit_only_unalt.csv', index=False)
    manip_df.to_csv('result/submit_only_manip.csv', index=False)

    pd.DataFrame(probas).to_csv(f'result/probas_{model_name}.csv', index=False)


if __name__ == '__main__':
    Fire(predict)
