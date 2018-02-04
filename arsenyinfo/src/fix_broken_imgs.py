from glob import glob
from os import remove

from fire import Fire
from tqdm import tqdm
import cv2
from glog import logger

from src.utils import get_img_attributes


def fix_and_check(x):
    with open(x, 'rb') as f:
        data = f.read()
        check_chars = data[-2:]
        if check_chars == b'\xff\xd9':
            return
    with open(x, 'wb') as f:
        f.write(data)

    img = cv2.imread(x)
    if img is None or img[-5:, :, :].std() == 0:
        logger.info(f'{x} is removed')
        remove(x)
        return

    q, soft = get_img_attributes(x)
    if q < 90:
        logger.info(f'{x} is removed: quality is {q}')
        remove(x)
        return

    if any(map(lambda x: x in soft, ('editor', 'adobe', 'aperture'))):
        logger.info(f'{x} is removed: software is {soft}')
        remove(x)
        return


def main(path):
    files = glob(path)
    for f in tqdm(files):
        fix_and_check(f)


if __name__ == '__main__':
    Fire(main)
