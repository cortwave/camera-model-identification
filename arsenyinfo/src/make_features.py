import logging
from glob import glob
from functools import partial

import cv2
import joblib
from keras.models import load_model, Model
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras import backend as K
from tqdm import tqdm
from fire import Fire

from src.predict import preprocess_input, batch_aug, _crop, get_test_files
from src.dataset import KaggleDataset
from src.aug import augment, center_crop

logging.getLogger('tensorflow').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S', )
logger = logging.getLogger(__name__)

N_FOLDS = 5


def make_crops(img, crop_size):
    img = center_crop(img, 512)
    crops = batch_aug([_crop(img, crop_size, x) for x in range(5)], safe=False)
    crops = preprocess_input(crops.astype('float32'))
    return crops


def dump(train, test, fold, model):
    joblib.dump(train, f'result/features/train_{model}_{fold}.h5')
    joblib.dump(test, f'result/features/test_{model}_{fold}.h5')
    return [], []


def main(model_name, shape):
    train = []
    test = []
    layer_name = 'global_average_pooling2d_1'

    load_model_ = partial(load_model, custom_objects={'relu6': relu6,
                                                      'DepthwiseConv2D': DepthwiseConv2D})

    for model_path in glob(f'result/models/{model_name}_*.h5'):
        model = load_model_(model_path)
        logger.info(f'Processing {model_path}')

        outputs = model.get_layer(layer_name).output
        feature_extractor = Model(inputs=model.inputs, outputs=outputs)

        for fold in range(N_FOLDS):
            if str(fold) in model_path.split('.')[0]:
                dataset = KaggleDataset(n_fold=fold,
                                        batch_size=32,
                                        size=shape,
                                        transform=preprocess_input,
                                        aug=partial(augment, expected_shape=shape),
                                        center_crop_size=0,
                                        train=False
                                        )
                train_data = dataset.data
                for fname, label in tqdm(train_data, desc='processing train'):
                    img = cv2.imread(fname)
                    crops = make_crops(img, crop_size=shape)
                    predictions = feature_extractor.predict(crops)

                    for pred in predictions:
                        train.append((fname.split('/')[-1], pred))

            for fname, crops in tqdm(get_test_files(shape=shape), desc='predicting test', total=2640):
                predictions = feature_extractor.predict(crops)
                for pred in predictions:
                    test.append((fname.split('/')[-1], pred))

            train, test = dump(train, test, fold, model_name)

        K.clear_session()


if __name__ == '__main__':
    Fire(main)
