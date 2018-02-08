from glob import glob

import numpy as np
import pandas as pd
from fire import Fire


def main(model_name):
    files = [pd.read_csv(x) for x in glob(f'result/probas_{model_name}*.csv')]
    df = pd.concat(files)
    df = df.groupby('fname').agg(np.sum).reset_index()
    df.to_csv(f'result/blended_probas_{model_name}.csv', index=False)
    df.describe()

    classes = ['HTC-1-M7', 'LG-Nexus-5x', 'Motorola-Droid-Maxx', 'Motorola-Nexus-6',
               'Motorola-X', 'Samsung-Galaxy-Note3', 'Samsung-Galaxy-S4', 'Sony-NEX-7',
               'iPhone-4s', 'iPhone-6']

    df['camera'] = 'tmp'

    ix = df.fname.str.contains('manip').index
    while np.any(df.iloc[ix]['camera'] == 'tmp'):
        for c in classes:
            idx = df[df.iloc[ix].camera == 'tmp'][c].argmax()
            df.set_value(col='camera', index=idx, value=c)

    ix = df.fname.str.contains('unalt').index
    while np.any(df.iloc[ix]['camera'] == 'tmp'):
        for c in classes:
            idx = df[df.iloc[ix].camera == 'tmp'][c].argmax()
            df.set_value(col='camera', index=idx, value=c)

    df['camera_predicted'] = df.apply(lambda x: x[classes].argmax(), axis=1)
    df['weird'] = df.camera_predicted == df.camera

    with open('result/weird.txt', 'w') as out:
        weird = df[~df.weird]
        for f in weird['fname']:
            out.write(f'{f}\n')

    unbalanced = df[['fname', 'camera_predicted']]
    unbalanced = unbalanced.rename(columns={'camera_predicted': 'camera'})
    unbalanced.to_csv(f'result/unbalanced_{model_name}.csv', index=False)

    df = df[['fname', 'camera']]
    df.to_csv(f'result/balanced_{model_name}.csv', index=False)

    unalt = df.fname.apply(lambda x: 'unalt' in x)
    manip = df.fname.apply(lambda x: 'manip' in x)

    unalt_df = df.copy()
    unalt_df['camera'][manip] = 'tmp'

    manip_df = df.copy()
    manip_df['camera'][unalt] = 'tmp'

    unalt_df.to_csv(f'result/{model_name}_only_unalt.csv', index=False)
    manip_df.to_csv(f'result/{model_name}_only_manip.csv', index=False)


if __name__ == '__main__':
    Fire(main)
