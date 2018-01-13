from torchvision import transforms
from io import BytesIO
from PIL import Image
import cv2
import numpy as np


def crop_and_flip():
    size = 350
    return transforms.Compose([
        RandomCrop(size * 2),
        RandomSelect([
            RandomResize((0.5, 2.0), 0.5),
            RandomGamma((0.7, 1.3), 0.5),
            RandomJPG((68, 92), 0.5),
        ]),
        RandomHFlip(),
        Denoise(),
        RandomCrop(size),
        transforms.ToTensor(),
    ])


def train_augm():
    return crop_and_flip()


def valid_augm():
    return crop_and_flip()


def test_augm():
    return transforms.Compose([
        RandomHFlip(),
        Denoise(),
        RandomCrop(350),
        transforms.ToTensor(),
    ])


class Denoise:
    def __init__(self):
        pass

    def __call__(self, img):
        h = np.random.uniform(2., 4.)
        hColor = np.random.uniform(2., 4.)
        denoised = cv2.fastNlMeansDenoisingColored(img, h=h, hColor=hColor)
        return img - denoised


class RandomSelect:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        t = np.random.choice(self.transforms)
        img = t(img)
        return img


class RandomJPG:
    def __init__(self, borders, prob):
        self.borders = borders
        self.prob = prob

    def __call__(self, img):
        if np.random.random() < self.prob:
            quality = np.random.randint(self.borders[0], self.borders[1])
            out = BytesIO()
            i = Image.fromarray(img)
            i.save(out, format='jpeg', quality=quality)
            out.seek(0)
            byte_img = out.read()
            data_bytes_io = BytesIO(byte_img)
            img = np.array(Image.open(data_bytes_io))
        return img


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        try:
            x = np.random.randint(0, img.shape[0] - self.size)
            y = np.random.randint(0, img.shape[1] - self.size)
        except Exception as e:
            print(img.shape)
            raise e
        return img[x:x + self.size, y:y + self.size, :]


class RandomHFlip:
    def __init__(self):
        pass

    def __call__(self, img):
        if np.random.random() < 0.5:
            return np.fliplr(img).copy()
        return img


class RandomResize:
    def __init__(self, borders, prob):
        self.borders = borders
        self.prob = prob

    def __call__(self, img):
        if np.random.random() < self.prob and img.shape[0] > 1024 and img.shape[1] > 1024:
            coeff = np.random.uniform(self.borders[0], self.borders[1])
            result = cv2.resize(img, dsize=None, fx=coeff, fy=coeff, interpolation=cv2.INTER_CUBIC)
            return result
        else:
            return img


class RandomGamma:
    def __init__(self, borders, prob):
        self.borders = borders
        self.prob = prob

    def __call__(self, img):
        if np.random.random() < self.prob:
            coeff = np.random.uniform(self.borders[0], self.borders[1])
            return self._adjust_gamma(img, coeff)
        else:
            return img

    @staticmethod
    def _adjust_gamma(image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)
