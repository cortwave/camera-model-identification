from dataloader import get_loaders
from dataset import TestDataset
from pathlib import Path
import random
import tqdm
from itertools import islice
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
import numpy as np
import torch
from datetime import datetime
import shutil
import json
from transforms import train_augm, valid_augm, test_augm
from pt_util import variable, long_tensor
from metrics import accuracy
from fire import Fire
from model_factory import get_model


def validate(model, criterion, valid_loader, validation_size, batch_size, iter_size):
    model.eval()
    losses = []
    accuracies = []
    batches_count = validation_size // batch_size
    valid_loader = islice(valid_loader, batches_count)
    for i, (inputs, targets) in tqdm.tqdm(enumerate(valid_loader), total=batches_count, desc="validation"):
        inputs = variable(inputs, volatile=True)
        targets = variable(targets)
        targets = long_tensor(targets)
        inputs0_chunks = inputs[0].chunk(iter_size)
        inputs1_chunks = inputs[1].chunk(iter_size)
        targets_chunks = targets.chunk(iter_size)
        loss = 0
        acc = 0
        for input1, input2, target in zip(inputs0_chunks, inputs1_chunks, targets_chunks):
            outputs = model(input1, input2)
            loss_iter = criterion(outputs, target)
            loss_iter /= batch_size
            loss += loss_iter.data[0]
            acc_iter = accuracy(outputs, target)[0]
            acc_iter /= iter_size
            acc += acc_iter.data[0]
        losses.append(loss)
        accuracies.append(acc)
    valid_loss = np.mean(losses)
    valid_acc = np.mean(accuracies)
    print('Valid loss: {:.4f}, acc: {:.4f}'.format(valid_loss, valid_acc))
    return {'valid_loss': valid_loss, 'valid_acc': valid_acc}


class Model(object):
    def train(self, architecture, fold, lr, batch_size, epochs, iter_size, epoch_size=None, validation_size=None,
              patience=4, optim="adam", ignore_prev_best_loss=False, cached_part=0.0, crop_central=False):
        train_loader, valid_loader, num_classes = get_loaders(batch_size,
                                                              train_transform=train_augm(),
                                                              valid_transform=valid_augm(),
                                                              n_fold=fold,
                                                              cached_part=cached_part,
                                                              crop_central=crop_central)
        validation_size = len(valid_loader) * batch_size
        model = get_model(num_classes, architecture)
        criterion = CrossEntropyLoss(size_average=False)

        self.ignore_prev_best_loss = ignore_prev_best_loss
        self.lr = lr
        self.model = model
        self.root = Path('../results/{}'.format(architecture))
        self.fold = fold
        self.optim = optim
        train_kwargs = dict(
            args=dict(iter_size=iter_size, n_epochs=epochs,
                      batch_size=batch_size, epoch_size=epoch_size),
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            valid_loader=valid_loader,
            validation_size=validation_size,
            patience=patience
        )
        self._train(**train_kwargs)

    def _init_optimizer(self):
        if self.optim == "adam":
            return Adam(self.model.parameters(), lr=self.lr, )
        elif self.optim == "sgd":
            return SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise Exception('Unknown optimizer: ', self.optim)

    def _init_files(self):
        if not self.root.exists():
            self.root.mkdir()
        self.log = self.root.joinpath('train_{}.log'.format(self.fold)).open('at', encoding='utf8')
        self.model_path = self.root / 'model_{}.pt'.format(self.fold)
        self.best_model_path = self.root / 'best-model_{}.pt'.format(self.fold)

    def _init_model(self):
        if self.model_path.exists():
            state = torch.load(str(self.model_path))
            self.epoch = state['epoch']
            self.step = state['step']
            if self.ignore_prev_best_loss:
                self.best_valid_loss = float('inf')
            else:
                self.best_valid_loss = state['best_valid_loss']
            self.model.load_state_dict(state['model'])
            print('Restored model, epoch {}, step {:,}'.format(self.epoch, self.step))
        else:
            self.epoch = 1
            self.step = 0
            self.best_valid_loss = float('inf')

    def _save_model(self, epoch):
        torch.save({
            'model': self.model.state_dict(),
            'epoch': epoch,
            'step': self.step,
            'best_valid_loss': self.best_valid_loss
        }, str(self.model_path))

    def _train(self,
               args,
               model,
               criterion,
               *,
               train_loader,
               valid_loader,
               validation_size,
               patience=2):
        lr = self.lr
        n_epochs = args['n_epochs']
        optimizer = self._init_optimizer()
        self._init_files()
        self._init_model()

        report_each = 10
        valid_losses = []
        lr_reset_epoch = self.epoch
        batch_size = args['batch_size']
        iter_size = args['iter_size']
        for epoch in range(self.epoch, n_epochs + 1):
            model.train()
            random.seed()
            tq = tqdm.tqdm(total=(args['epoch_size'] or
                                  len(train_loader) * batch_size))
            tq.set_description('Epoch {}, lr {}'.format(epoch, lr))
            losses = []
            tl = train_loader
            epoch_loss = 0
            if args['epoch_size']:
                tl = islice(tl, args['epoch_size'] // batch_size)
            try:
                mean_loss = 0
                batches_count = 0
                for i, (inputs, targets) in enumerate(tl):
                    batches_count += 1
                    inputs, targets = variable(inputs), variable(targets)
                    targets = long_tensor(targets)
                    inputs_0_chunks = inputs[0].chunk(iter_size)
                    inputs_1_chunks = inputs[1].chunk(iter_size)
                    targets_chunks = targets.chunk(iter_size)
                    optimizer.zero_grad()

                    iter_loss = 0
                    for input1, input2, target in zip(inputs_0_chunks, inputs_1_chunks, targets_chunks):
                        outputs = model(input1, input2)
                        loss = criterion(outputs, target)
                        loss /= batch_size
                        iter_loss += loss.data[0]
                        loss.backward()
                    optimizer.step()
                    self.step += 1
                    tq.update(batch_size)
                    epoch_loss += iter_loss
                    losses.append(iter_loss)
                    mean_loss = np.mean(losses[-report_each:])
                    tq.set_postfix(loss='{:.3f}'.format(mean_loss))
                    if i and i % report_each == 0:
                        self._write_event(loss=mean_loss)
                epoch_loss /= batches_count
                self._write_event(loss=mean_loss)
                tq.close()
                self._save_model(epoch + 1)
                valid_metrics = validate(model, criterion, valid_loader, validation_size, batch_size, iter_size)
                self._write_event(**valid_metrics)
                valid_loss = valid_metrics['valid_loss']
                valid_losses.append(valid_loss)
                if valid_loss < self.best_valid_loss:
                    print("Best validation loss improved from {} to {}".format(self.best_valid_loss, valid_loss))
                    self.best_valid_loss = valid_loss
                    shutil.copy(str(self.model_path), str(self.best_model_path))
                elif patience and epoch - lr_reset_epoch > patience and min(
                        valid_losses[-patience:]) > self.best_valid_loss:
                    lr /= 10
                    if lr < 1e-8:
                        exit(0)
                    lr_reset_epoch = epoch
                    optimizer = self._init_optimizer()
            except KeyboardInterrupt:
                tq.close()
                print('Ctrl+C, saving snapshot')
                self._save_model(epoch)
                print('done.')
                break
        return

    def _write_event(self, **data):
        data['step'] = self.step
        data['dt'] = datetime.now().isoformat()
        self.log.write(json.dumps(data, sort_keys=True))
        self.log.write('\n')
        self.log.flush()

    def predict(self, architecture, fold, tta=5, mode='submit', name="sub"):
        test_dataset = TestDataset(transform=test_augm())
        model = get_model(num_classes=test_dataset.num_classes, architecture=architecture)
        state = torch.load('../results/{}/best-model_{}.pt'.format(architecture, fold))
        model.load_state_dict(state['model'])
        model.eval()
        if mode == 'submit':
            with open('../results/{}/{}_{}.csv'.format(architecture, name, fold), "w") as f:
                f.write("fname,camera\n")
                for idx in tqdm.tqdm(range(len(test_dataset))):
                    images = torch.stack([test_dataset[idx][0] for _ in range(tta)])
                    images = variable(images)
                    pred = model(images).data.cpu().numpy()
                    pred = np.sum(pred, axis=0)
                    fname = test_dataset[idx][1]
                    label = np.argmax(pred, 0)
                    camera_model = test_dataset.inverse_dict[label]
                    f.write('{},{}\n'.format(fname, camera_model))
        else:
            def softmax(x):
                """Compute softmax values for each sets of scores in x."""
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum(axis=0)

            with open('../results/{}/{}_{}_prob.csv'.format(architecture, name, fold), "w") as f:
                for idx in tqdm.tqdm(range(len(test_dataset))):
                    images = torch.stack([test_dataset[idx][0] for _ in range(tta)])
                    images = variable(images)
                    pred = model(images).data.cpu().numpy()
                    pred = np.array([softmax(x) for x in pred])
                    pred = np.sum(pred, axis=0) / len(pred)
                    fname = test_dataset[idx][1]
                    probas = ','.join([str(x) for x in pred])
                    f.write('{},{}\n'.format(fname, probas))


if __name__ == '__main__':
    Fire(Model)
