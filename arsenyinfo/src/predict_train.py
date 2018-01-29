import logging
from glob import glob
from functools import partial

import cv2
import pandas as pd
from keras.models import load_model
from keras import backend as K
from tqdm import tqdm
from fire import Fire

from src.predict import preprocess_input, batch_aug, _crop
from src.dataset import KaggleDataset
from src.aug import augment, center_crop

logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S', )
logger = logging.getLogger(__name__)

N_FOLDS = 5


def make_crops(img):
    img = center_crop(img, 512)
    crops = batch_aug([_crop(img, 384, x) for x in range(5)], safe=False)
    crops = preprocess_input(crops.astype('float32'))
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
                        d['fold'] = fold
                        train.append(d)
        K.clear_session()


    pd.DataFrame(train).to_csv(f'result/train_{model_name}.csv', index=False)


if __name__ == '__main__':
    Fire(main)
