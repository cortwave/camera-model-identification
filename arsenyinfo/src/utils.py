import logging
import subprocess

logging.basicConfig(level=logging.INFO,
                    format='%(levelname)s: %(name)s: %(message)s (%(asctime)s; %(filename)s:%(lineno)d)',
                    datefmt="%Y-%m-%d %H:%M:%S", )
logger = logging.getLogger(__name__)


def get_img_attributes(fname):
    # ToDo: this should be refactored to be faster
    s = subprocess.run([f'identify', '-verbose', f'{fname}'],
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       )

    s = s.stdout.decode().split('\n')

    try:
        quality = int(list(filter(lambda x: 'Quality' in x, s))[0].split(': ')[-1])
    except Exception:
        logger.exception(f'Can not parse {fname} quality')
        quality = 0

    try:
        soft = [x for x in s if 'Software' in x]
        if soft:
            soft = soft[0].split(': ')[-1].lower()
        else:
            soft = ''
    except Exception:
        logger.exception(f'Can not parse {fname} software')
        soft = ''

    return quality, soft
