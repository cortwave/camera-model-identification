from urllib import request
import os
from tqdm import tqdm
import shutil
from joblib import Parallel, delayed


def load_image(address, name):
    basename = os.path.basename(address)
    save_path = f'../../extra_data/flickr_images/{name}/{basename}'
    if not os.path.exists(save_path):
        request.urlretrieve(address, save_path)


if __name__ == '__main__':
    models_dict = {
         'samsung_note3': 'Samsung-Galaxy-Note3', 
         'moto_maxx': 'Motorola-Droid-Maxx', 
         'moto_x': 'Motorola-X', 
         'samsung_s4': 'Samsung-Galaxy-S4', 
         'htc_m7': 'HTC-1-M7', 
         'sony_nex7': 'Sony-NEX-7', 
         'iphone_6': 'iPhone-6', 
         'nexus_6': 'Motorola-Nexus-6', 
         'iphone_4s': 'iPhone-4s', 
         'nexus_5x': 'LG-Nexus-5x'
    }
    for folder in ['flickr_images']:
        for extra_name, real_name in models_dict.items():
            print(extra_name, real_name)
            with open(f'../../extra_data/{folder}/{extra_name}/urls_final', 'r') as f:
                Parallel(n_jobs=4)(delayed(load_image)(line.strip(), extra_name) for line in tqdm(f.readlines()))

    with open(f'../../extra_data/flickr_images/good_jpgs', 'r') as f:
        for line in f.readlines():
            line = line.strip()
            _, name, base_name = line.split('/')
            real_name = models_dict[name]
            shutil.copy(f'../../extra_data/flickr_images/{name}/{base_name}', f'../../data/train/{real_name}/{base_name}')

