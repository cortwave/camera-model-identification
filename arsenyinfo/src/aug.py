from io import BytesIO
from functools import partial

import numpy as np
import cv2
from PIL import Image

from src.utils import logger


def resize(x, shape, factor=None):
    if factor is None:
        factor = np.random.randint(51, 201) / 100.
    new_shape = max(shape, int(x.shape[0] * factor))

    big = cv2.resize(x, (new_shape, new_shape), interpolation=cv2.INTER_CUBIC)
    if new_shape == shape:
        return big
    return crop(big, target_shape=shape)


def center_crop(x, size):
    a, b, _ = x.shape

    if any((a < size, b < size)):
        return x

    a = (a - size) // 2
    b = (b - size) // 2

    x = x[a:-a, b:-b, :]
    return x


def crop(img, target_shape):
    a, b, _ = img.shape
    if any((a < target_shape, b < target_shape)):
        return img
    try:
        w = np.random.randint(0, a - target_shape)
        h = np.random.randint(0, b - target_shape)
    except:
        logger.exception(f'{a}, {b}, {target_shape}')
    img = img[w: w + target_shape, h: h + target_shape, ...]
    return img


def jpg_compress(x, quality=None):
    if quality is None:
        quality = np.random.randint(70, 91)
    x = Image.fromarray(x)
    out = BytesIO()
    x.save(out, format='jpeg', quality=quality)
    x = Image.open(out)
    return np.array(x)


def gamma_correction(x, gamma=None):
    if gamma is None:
        gamma = np.random.randint(80, 121) / 100.
    x = x.astype('float32') / 255.
    x = np.power(x, gamma)
    return x * 255


def augment(x, expected_shape, safe=False):
    if safe:
        augs = (np.fliplr,
                np.flipud,
                partial(np.rot90, k=1),
                partial(np.rot90, k=3),
                None)
    else:
        augs = (partial(resize, shape=expected_shape),
                jpg_compress,
                gamma_correction,
                np.fliplr,
                np.flipud,
                partial(np.rot90, k=1),
                partial(np.rot90, k=3),
                None)
    f = np.random.choice(augs)

    if f is not augs[0]:
        x = crop(x, expected_shape)
    if f is not None:
        return f(x)
    return x
