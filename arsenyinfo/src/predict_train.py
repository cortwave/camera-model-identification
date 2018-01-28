import logging
from glob import glob
from functools import partial

import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from joblib import Parallel, delayed
from tqdm import tqdm
from fire import Fire

from src.predict import preprocess_input
from src.dataset import KaggleDataset
from src.aug import augment

logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S', )
logger = logging.getLogger(__name__)

N_FOLDS = 5

POOL = Parallel(n_jobs=4, backend='multiprocessing')


def make_crops(img):
    crops = np.array(POOL(delayed(augment)(img, expected_shape=384) for _ in range(32)), dtype='float32')
    crops = preprocess_input(crops)
    return crops


def main(model_name):
    train = []
    # layer_name = 'global_average_pooling2d_1'

    for model_path in glob(f'result/models/{model_name}_*.h5'):
        model = load_model(model_path)
        logger.info(f'Processing {model_path}')

        # outputs = model.get_layer(layer_name).output
        # feature_extractor = Model(inputs=model.inputs, outputs=outputs)

        for fold in range(N_FOLDS):
            if str(fold) in model_path.split('.')[0]:
                shape = 384
                dataset = KaggleDataset(n_fold=fold,
                                        batch_size=32,
                                        size=shape,
                                        transform=preprocess_input,
                                        aug=partial(augment, expected_shape=shape),
                                        center_crop_size=0,
                                        train=False
                                        )
                train_data = dataset.data
                classes = dataset.classes
                for fname, label in tqdm(train_data, desc='processing'):
                    img = cv2.imread(fname)
                    crops = make_crops(img)
                    predictions = model.predict(crops)

                    for pred in predictions:
                        d = {k: v for k, v in zip(classes, pred)}
                        d['label'] = classes[int(label)]
                        d['model'] = model_path.split('/')[-1]
                        train.append(d)

    pd.DataFrame(train).to_csv(f'result/train_{model_name}.csv', index=False)


if __name__ == '__main__':
    Fire(main)
