from urllib import request
import os
from time import sleep
from glob import glob
from glog import logger

from tqdm import tqdm
from joblib import Parallel, delayed
from fire import Fire


def load_image(address, folder_name):
    i = 1
    success = False
    while not success:
        try:
            _load_image(address, folder_name)
            success = True
        except Exception:
            logger.exception(f'{i} retries failed for {address}')
            i += 1
            sleep(i)


def _load_image(address, folder_name):
    basename = os.path.basename(address)
    save_path = f'data/train/{folder_name}/{basename}'
    if not os.path.exists(save_path):
        request.urlretrieve(address, save_path)


def main(start):
    url_lists = glob(start)
    for l in url_lists:
        name = l.split('/')[-1].split('.')[0]
        print(name)
        with open(l) as f:
            Parallel(n_jobs=4)(delayed(load_image)(line.strip(), name) for line in tqdm(f.readlines()))


if __name__ == '__main__':
    Fire(main)
