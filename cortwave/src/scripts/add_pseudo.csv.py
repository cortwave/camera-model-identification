import pandas as pd
import shutil


if __name__ == '__main__':
    pseudo = pd.read_csv('../../data/pseudo.csv')
    for fname, camera in zip(pseudo.fname.values, pseudo.camera.values):
        shutil.copy('../../data/test/{}'.format(fname), '../../data/train/{}/{}'.format(camera, fname
                                                                                        ))