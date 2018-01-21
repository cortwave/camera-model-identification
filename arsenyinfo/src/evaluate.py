from functools import partial

from keras.models import load_model
from keras.applications.inception_resnet_v2 import preprocess_input
from fire import Fire

from src.dataset import KaggleDataset
from src.aug import jpg_compress, gamma_correction, resize
from src.utils import logger


def fit_model(model, n_fold=1):
    model = load_model(f'result/models/kaggle_{model}_crop_{n_fold}.h5')

    shape = 384

    n_fold = int(n_fold)

    for aug in (gamma_correction, jpg_compress, lambda x: x, partial(resize, shape=shape * 2)):
        aug_ = lambda x: resize(aug(x), shape)

        val = KaggleDataset(n_fold=n_fold,
                            transform=preprocess_input,
                            batch_size=16,
                            train=False,
                            size=shape,
                            aug=aug_,
                            center_crop_size=0)

        loss, acc = model.evaluate_generator(val, steps=10)
        logger.info(f'{aug} - {acc}')


if __name__ == '__main__':
    Fire(fit_model)
