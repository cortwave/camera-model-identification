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


def load_direct(camera_name, url):
    basename = os.path.basename(url)
    save_path = f'../../data/train/{camera_name}/{basename}'
    if not os.path.exists(save_path):
        request.urlretrieve(url, save_path)


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
        missed = 0
        for line in tqdm(f.readlines(), desc='images copying...'):
            line = line.strip()
            _, name, base_name = line.split('/')
            real_name = models_dict[name]
            dest = f'../../data/train/{real_name}/{base_name}'
            if not os.path.exists(dest):
                try:
                    shutil.copy(f'../../extra_data/flickr_images/{name}/{base_name}', dest)
                except Exception:
                    print('missed file: ', line)
                    missed += 1
        print('totally missed', missed)

    for camera in models_dict.values():
        with open(f'../../extra_data/bes_external_data/{camera}/{camera}.csv', 'r') as f:
            Parallel(n_jobs=4)(delayed(load_direct)(camera, line.strip()) for line in tqdm(f.readlines(), desc=camera))
