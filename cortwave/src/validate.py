from fire import Fire
import torch
from pt_util import variable
import numpy as np
import tqdm
import pandas as pd

from transforms import test_augm
from model_factory import get_model
from util import softmax
from dataset import InternValidDataset


def main(architecture,
         folds,
         tta):
    test_dataset = InternValidDataset(transform=test_augm())
    labels = None
    for fold in folds:
        model = get_model(num_classes=test_dataset.num_classes, architecture=architecture)
        state = torch.load('../results/{}/best-model_{}.pt'.format(architecture, fold))
        model.load_state_dict(state['model'])
        model.eval()
        labels = []
        with open('../results/{}/{}_valid_prob.csv'.format(architecture, fold), "w") as f:
            for idx in tqdm.tqdm(range(len(test_dataset))):
                best_conf = 0
                best_pred = None
                for rot in range(4):
                    test_dataset.rot = rot
                    in1 = []
                    in2 = []
                    for _ in range(tta):
                        x = test_dataset[idx][0]
                        in1.append(x[0])
                        in2.append(x[1])
                    in1 = variable(torch.stack(in1))
                    in2 = variable(torch.stack(in2))
                    pred = model(in1, in2).data.cpu().numpy()
                    pred = np.array([softmax(x) for x in pred])
                    pred = np.sum(pred, axis=0) / len(pred)
                    if np.max(pred) > best_conf:
                        best_conf = np.max(pred)
                        best_pred = pred
                labels.append(test_dataset[idx][1])
                probas = ','.join([str(x) for x in best_pred])
                f.write('{}\n'.format(probas))

    dfs = [pd.read_csv('../results/{}/{}_valid_prob.csv'.format(architecture, i), header=None) for i in folds]
    classes = ['HTC-1-M7', 'LG-Nexus-5x', 'Motorola-Droid-Maxx',
               'Motorola-Nexus-6', 'Motorola-X', 'Samsung-Galaxy-Note3',
               'Samsung-Galaxy-S4', 'Sony-NEX-7', 'iPhone-4s', 'iPhone-6']
    for df in dfs:
        df.columns = classes
    df = dfs[0].copy()
    for i in np.arange(1, len(folds)):
        df[classes] += dfs[i][classes]
    df[classes] /= len(folds)
    matched = 0
    for i in np.arange(len(test_dataset)):
        pred = df[classes].iloc[i].values.argmax()
        real = labels[i]
        if pred == real:
            matched += 1
    print('accuracy = {}'.format(matched / len(test_dataset)))


if __name__ == '__main__':
    Fire(main)
