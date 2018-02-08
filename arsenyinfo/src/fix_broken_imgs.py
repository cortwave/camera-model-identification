from glob import glob
from os import remove

from fire import Fire
from tqdm import tqdm
import cv2
from joblib import Parallel, delayed

from src.utils import get_img_attributes


def fix_thread_safe(x):
    with open(x, 'rb') as f:
        data = f.read()
        check_chars = data[-2:]
        if not check_chars == b'\xff\xd9':
            with open(x, 'wb') as f:
                f.write(data)

    img = cv2.imread(x)
    if img is None or img[-5:, :, :].std() == 0:
        remove(x)
        return
    if img.shape[0] < 1000 or img.shape[1] < 1000:
        remove(x)
        return


def fix_thread_unsafe(x):
    try:
        q, soft = get_img_attributes(x)
    except:
        print('broken')
        return
    if q < 93:
        remove(x)
        return

    if any(map(lambda x: x in soft, ('editor', 'adobe', 'aperture', 'paint'))):
        remove(x)
        return


def main(path):
    files = glob(path)
    Parallel(n_jobs=-1)(delayed(fix_thread_safe)(x) for x in tqdm(files))
    for x in tqdm(files):
        fix_thread_unsafe(x)


if __name__ == '__main__':
    Fire(main)
