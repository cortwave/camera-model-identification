from torchvision import transforms
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from skimage.transform import rotate

size = 128


def transform(img, manip):
    ops = []
    if not manip:
        ops.append(
            RandomSelect([
                RandomResize((0.5, 0.8, 1.5, 2.0), 0.5),
                RandomGamma((0.8, 1.2), 0.5),
                RandomJPG((70, 90), 0.5),
            ])
        )
    for o in [RandomCrop(size), transforms.ToTensor()]:
        ops.append(o)
    ops = transforms.Compose(ops)
    return ops(img)


def train_augm():
    return transform


def valid_augm():
    return transform


def test_augm():
    return transforms.Compose([
        RandomCrop(size),
        transforms.ToTensor(),
    ])


class ExtractFFTNoise:
    def __init__(self):
        pass

    def __call__(self, img):
        img -= cv2.GaussianBlur(img, (3, 3), 0)
        img = np.stack([np.fft.fftshift(np.fft.fft2(img[:, :, c])) for c in range(3)], axis=-1)
        return img


class ExtractNoise:
    def __init__(self):
        pass

    def __call__(self, img):
        kernel_filter = 1 / 12. * np.array([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]
        ])
        return cv2.filter2D(img, -1, kernel_filter)


class Denoise:
    def __init__(self):
        pass

    def __call__(self, img):
        h = np.random.uniform(2., 4.)
        hColor = np.random.uniform(2., 4.)
        denoised = cv2.fastNlMeansDenoisingColored(img, h=h, hColor=hColor)
        return img - denoised


class RandomSelect:
    def __init__(self, ops):
        self.ops = ops

    def __call__(self, img):
        t = np.random.choice(self.ops)
        img = t(img)
        return img


class RandomJPG:
    def __init__(self, borders, prob):
        self.borders = borders
        self.prob = prob

    def __call__(self, img):
        quality = np.random.choice(self.borders)
        quality = int(quality)
        if np.random.random() < self.prob:
            out = BytesIO()
            i = Image.fromarray(img)
            i.save(out, format='jpeg', quality=quality)
            out.seek(0)
            byte_img = out.read()
            data_bytes_io = BytesIO(byte_img)
            img = np.array(Image.open(data_bytes_io))
        return img


class RandomCrop:
    def __init__(self, size, strict=True):
        self.size = size
        self.strict = strict

    def __call__(self, img):
        if (img.shape[0] < self.size or img.shape[1] < self.size) and not self.strict:
            return img
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


class RandomVFlip:
    def __init__(self):
        pass

    def __call__(self, img):
        if np.random.random() < 0.5:
            return np.flipud(img).copy()
        return img


class RandomRotate:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, img):
        angle = np.random.choice(self.angles)
        img = rotate(img, angle)
        return img


class RandomResize:
    def __init__(self, borders, prob):
        self.borders = borders
        self.prob = prob

    def __call__(self, img):
        coeff = np.random.choice(self.borders)
        if np.random.random() < self.prob and img.shape[0] * coeff > size and img.shape[1] * coeff > size:
            result = cv2.resize(img, dsize=None, fx=coeff, fy=coeff, interpolation=cv2.INTER_CUBIC)
            return result
        else:
            return img


class RandomGamma:
    def __init__(self, borders, prob):
        self.borders = borders
        self.prob = prob

    def __call__(self, img):
        coeff = np.random.choice(self.borders)
        if np.random.random() < self.prob:
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
