import torch.utils.data as data

from dataset import Dataset, TestDataset


def get_test_loader(batch_size, transform=None):
    test_dataset = TestDataset(transform)
    test_loader = data.DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=6)
    return test_loader


def get_valid_loader(n_fold, batch_size, transform):
    dataset = Dataset(n_fold, transform, train=False)
    dataset = data.DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=6)
    return dataset


def get_loaders(batch_size,
                n_fold,
                train_transform=None,
                valid_transform=None,
                cached_part=0.0,
                crop_central=False):
    train_dataset = Dataset(n_fold, transform=train_transform, train=True, cached_part=cached_part, crop_central=crop_central)
    train_loader = data.DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=6,
                                   pin_memory=True)

    valid_dataset = Dataset(n_fold, transform=valid_transform, train=False, cached_part=cached_part, crop_central=crop_central)
    valid_loader = data.DataLoader(valid_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=6,
                                   pin_memory=True)
    return train_loader, valid_loader, train_dataset.num_classes