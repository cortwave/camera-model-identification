from torchvision import transforms
from io import BytesIO
from PIL import Image
import cv2
import numpy as np


def crop_and_flip():
    return transforms.Compose([
        RandomJPG([70, 90]),
        RandomResize([0.5, 0.8, 1.5, 2.0], 0.25),
        RandomGamma([0.8, 1.2], 0.25),
        RandomHFlip(),
        RandomCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def train_augm():
    return crop_and_flip()


def valid_augm():
    return crop_and_flip()


def test_augm():
    return transforms.Compose([
        RandomHFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class RandomJPG:
    def __init__(self, quality):
        self.quality = quality

    def __call__(self, img):
        if np.random.random() < 0.5:
            quality = np.random.choice(self.quality)
            out = BytesIO()
            i = Image.fromarray(img)
            i.save(out, format='jpeg', quality=quality)
            out.seek(0)
            byte_img = out.read()

            # Non test code
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
    def __init__(self, coeffs, prob):
        self.coeffs = coeffs
        self.prob = prob

    def __call__(self, img):
        if np.random.random() < self.prob and img.shape[0] > 1024 and img.shape[1] > 1024:
            coeff = np.random.choice(self.coeffs)
            result = cv2.resize(img, dsize=None, fx=coeff, fy=coeff, interpolation=cv2.INTER_CUBIC)
            return result
        else:
            return img


class RandomGamma:
    def __init__(self, coeffs, prob):
        self.coeffs = coeffs
        self.prob = prob

    def __call__(self, img):
        if np.random.random() < self.prob:
            coeff = np.random.choice(self.coeffs)
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
