import pandas as pd
from os import remove
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
import subprocess
import numpy as np
import os
import cv2

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(name)s: %(message)s (%(asctime)s; %(filename)s:%(lineno)d)',
                    datefmt="%Y-%m-%d %H:%M:%S", )
logger = logging.getLogger(__name__)


def get_img_attributes(fname):
    try:
        s = subprocess.run([f'identify', '-verbose', f'{fname}'],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE,
                           )

        s = s.stdout.decode().split('\n')
    except UnicodeDecodeError:
        return 0, None

    try:
        quality = int(list(filter(lambda x: 'Quality' in x, s))[0].split(': ')[-1])
    except Exception:
        print(s)
        logger.exception(f'Can not parse {fname} quality')
        quality = 0

    try:
        soft = [x for x in s if 'Software' in x]
        if soft:
            soft = soft[0].split(': ')[-1]
        else:
            soft = ''
    except Exception:
        logger.exception(f'Can not parse {fname} software')
        soft = ''

    return quality, soft


def fix_and_check(x):
    if not os.path.exists(x):
        return True
    with open(x, 'rb') as f:
        data = f.read()
        check_chars = data[-2:]
        if not check_chars == b'\xff\xd9':
            with open(x, 'wb') as f:
                f.write(data)

    img = cv2.imread(x)
    if img is None or img[-5:, :, :].std() == 0:
        remove(x)
        return True

    q, soft = get_img_attributes(x)
    if q < 90:
        remove(x)
        return True

    if any(map(lambda x: x in soft, ('editor', 'adobe', 'aperture'))):
        remove(x)
        return True
    return False


if __name__ == '__main__':
    for i in range(5):
        df = pd.read_csv(f'../../data/fold_{i}.csv', header=None)
        images = df[0].values
        checks = [fix_and_check(image) for image in tqdm(images, desc=f'fold_{i}')]
        print('totally removed: ', np.sum(checks))
