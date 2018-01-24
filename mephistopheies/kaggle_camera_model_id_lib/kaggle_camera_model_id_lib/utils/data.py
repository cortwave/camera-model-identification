import numpy as np
from torchvision.datasets import ImageFolder
from skimage.transform import resize
import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import LabelEncoder


def patch_quality_dich(img_np, alpha=0.7, beta=4, gamma=np.log(0.01)):
    q = 0
    if img_np.max() > 1:
        img_np = img_np.copy()/255.0
    for ci in range(img_np.shape[2]):
        m = img_np[:, :, ci].mean()
        s = img_np[:, :, ci].std()
        q += alpha*beta*m*(1 - m) + (1 - alpha)*(1 - np.exp(gamma*s))    
    return q/img_np.shape[2]

def n_random_crops(img_np, n_crops, crop_size, f_score=None):
    x = np.random.randint(0, img_np.shape[0] - crop_size, size=n_crops)
    y = np.random.randint(0, img_np.shape[1] - crop_size, size=n_crops)
    for i in range(n_crops):
        img = img_np[x[i]:(x[i] + crop_size), y[i]:(y[i] + crop_size), :]
        if f_score is None:
            yield img
        else:
            yield img, f_score(img)


def n_pseudorandom_crops(img_np, crop_size, n_crops, n_search_space, f_score):
    res = sorted(n_random_crops(img_np, n_search_space, crop_size, f_score=f_score), key=lambda t: t[1], reverse=True)
    return [img for img, _ in res[:n_crops]]


def NCrops(img, crop_size=128, step=64, x_step=None, y_step=None):
    images = []
    if step is not None:
        x_step = step
        y_step = step
    for i in range(int(np.floor(1 + (img.shape[0] - crop_size)/y_step))):
        for j in range(int(np.floor(1 + (img.shape[1] - crop_size)/x_step))):
            images.append(img[i*y_step:(i*y_step + crop_size), 
                              j*x_step:(j*x_step + crop_size), :])
    return images

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


class ImageList(Dataset):
    
    def __init__(self, fname, transform=None, target_transform=None):
        with open(fname, 'r') as f:
            files = f.readlines()

        classes, files = zip(*map(lambda s: (s.strip().split('/')[-2], s.strip()), files))
        classes = list(set(classes))
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        imgs = list(map(lambda s: (s, class_to_idx[s.split('/')[-2]]), files))
        
        if len(imgs) == 0:
            raise(RuntimeError('Found 0 images in: ' + fname))
            
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform      
        
    def loader(self, path):
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
            
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    

class ImageListExFiles(ImageList):
    
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, path
    

def make_dataset_npz(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if os.path.basename(fname).split('.')[-1].lower() == 'npz':
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class NpzFolder(ImageFolder):
    
    def loader(self, fname):
        with np.load(fname) as f:
            return f['data']
    
    def __init__(self, root, transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset_npz(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError('Found 0 npz files'))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        
        
class NpzFolderExFiles(ImageFolder):
    
    def loader(self, fname):
        with np.load(fname) as f:
            return f['data']
    
    def __init__(self, root, transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset_npz(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError('Found 0 npz files'))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, path
        

def make_dataset_tif(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if os.path.basename(fname).split('.')[-1].lower() == 'tif':
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images
        
        
class TifFolder(ImageFolder):   
    
    def loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    
    def __init__(self, root, transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset_tif(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError('Found 0 tif files'))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        
        
class TifFolderExFiles(ImageFolder):   
    
    def loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    
    def __init__(self, root, transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset_tif(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError('Found 0 tif files'))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target, path
    
    
    
class MultiDataset(Dataset):
    
    def __init__(self, datasets):
        self.img2ds = {}
        self.imgs = []
        self.class_to_idx = datasets[0].class_to_idx
        self.sizes = []
        for ds in datasets:
            self.sizes.append(len(ds))
            for ix, (img_name, c) in enumerate(ds.imgs):
                self.imgs.append((img_name, c))
                self.img2ds[len(self.img2ds)] = (ds, ix)                
                
    def __len__(self):
        return sum(self.sizes)
    
    def __getitem__(self, index):
        ds, ix = self.img2ds[index]
        return ds[ix]