from glob import glob
from os import remove

from fire import Fire
from tqdm import tqdm
import cv2


def fix_and_check(x):
    with open(x, 'rb') as f:
        data = f.read()
        check_chars = data[-2:]
        if check_chars == b'\xff\xd9':
            return
    with open(x, 'wb') as f:
        f.write(data)

    img = cv2.imread(x)
    if img is None or img[-10:, :, :].std() == 0:
        print(f'{x} is removed')
        remove(x)


def main(path):
    files = glob(path)
    for f in tqdm(files):
        fix_and_check(f)


if __name__ == '__main__':
    Fire(main)