import sys

import requests
import lxml.html as lh
from glog import logger
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

host = 'http://vfl.ru'
cameras = {'iPhone-4s': 'http://vfl.ru/fotos/camera/1676_ID.html',
           'Motorola-Droid-Maxx': 'http://vfl.ru/fotos/camera/4477_ID.html',
           'iPhone-6': 'http://vfl.ru/fotos/camera/6634_ID.html',
           'Motorola-Nexus-6': 'http://vfl.ru/fotos/camera/7971_ID.html'
           }

s = requests.Session()


def _fetch(url):
    return lh.document_fromstring(s.get(url).content)


def wrapper(f):
    def inner_f(*args, **options):
        try:
            return f(*args, **options)
        except KeyboardInterrupt:
            logger.info('Parsing stopped with KeyboardInterrupt')
            sys.exit()
        except Exception:
            logger.exception('Parsing failed at {}'.format(args))

    return inner_f


@wrapper
def parse_page(url):
    page = _fetch(url)
    photos = page.xpath('//*[@id="content"]/div/div[*]/a')
    photos = list(map(lambda x: host + x.get('href'), photos))
    return photos


@wrapper
def parse_photo(url):
    page = _fetch(url)
    src = page.xpath('//*[@id="img_foto"]')[0].get('src')
    meta = page.xpath('//*[@id="r"]')[0].text_content().split('\n')
    soft = [x for x in meta if 'Программа' in x]
    if soft:
        soft = soft[0].split(': ')[-1]
    return src, soft


def run():
    for c in cameras:
        logger.info(f'Working with {c}')
        photos = []
        for i in range(10):
            url = cameras[c].replace('ID', str(i + 1))
            photos += parse_page(url)
        logger.info(f'Page URLs parsed for {c}: {len(photos)}')

        for img in tqdm(photos):
            parsed = parse_photo(img)
            if parsed:
                src, soft = parsed
                yield {'camera': c, 'url': src, 'soft': soft}


def main():
    df = pd.DataFrame(list(run()))
    df.to_csv('vfl_photos.csv', index=False)


if __name__ == '__main__':
    main()
