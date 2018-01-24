import io
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from imgaug import augmenters as iaa
import torchvision.transforms as transforms


def hsv_convert(img):
    if Image.isImageType(img):
        img = np.array(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)


def equalize_v_hist(img):
    is_pil = False
    if Image.isImageType(img):
        img = np.array(img)
        is_pil = True
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
    if is_pil:
        return Image.fromarray(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB))
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)


def jpg_compress(x, quality=(70, 90)):
    if not Image.isImageType(x):
        x = Image.fromarray(x)    
    out = BytesIO()
    x.save(out, format='jpeg', 
           quality=np.random.randint(quality[0], quality[1]))
    x = Image.open(out)
    return x


def scale_crop_pad(img, factor):
    size = img.size[0]
    new_size = int(size*factor)
    img = img.resize((new_size, new_size), Image.BICUBIC)
    if new_size < size:
        img = iaa.Pad(px=(size - new_size, size - new_size, 0, 0), 
                      pad_mode='symmetric', keep_size=False).augment_image(np.array(img))
        return Image.fromarray(img)
    return transforms.CenterCrop(size)(img)


def gamma_correction(img, gamma):
    return Image.fromarray((((np.array(img)/255.0)**(1/gamma))*255).astype(np.uint8))

