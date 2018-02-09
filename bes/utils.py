import os
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import threading
import cv2
import random
import math

import keras
from keras.layers import Dense,Dropout, Input, concatenate
from keras.models import Model
from keras.utils import to_categorical
import keras.optimizers as optimizers

from multiprocessing import Pool
pool = Pool(processes=12)

# 384 earlier
PICS_WIDTH = 384
PICS_HEIGHT = 384
NUM_CATEGS = 10

AUGMENTATION = 'PseudoLabeling'

CROP_SIZE = 512

MODEL_LOSS = 'categorical_crossentropy'
MODEL_METRIC = 'categorical_accuracy'

def InitialiazeModel(head_only,weights,model_name,lr):
    
    if model_name == 'VGG19':
        from keras.applications.vgg19 import VGG19
        base_model = VGG19(include_top=False, weights='imagenet',
                      input_shape=(PICS_WIDTH, PICS_HEIGHT, 3), pooling = 'avg')
    elif model_name == 'VGG16':
        from keras.applications.vgg16 import VGG16
        base_model = VGG16(include_top=False, weights='imagenet',
                      input_shape=(PICS_WIDTH, PICS_HEIGHT, 3), pooling = 'avg')
    elif model_name == 'MobileNet':
        from keras.applications.mobilenet import MobileNet
        base_model = MobileNet(include_top=False, weights='imagenet',
                      input_shape=(PICS_WIDTH, PICS_HEIGHT, 3), pooling = 'avg')
    elif model_name == 'ResNet50':
        from keras.applications.resnet50 import ResNet50
        base_model = ResNet50(include_top=False, weights='imagenet',
                      input_shape=(PICS_WIDTH, PICS_HEIGHT, 3), pooling = 'avg')
    elif model_name == 'Xception':
        from keras.applications.xception import Xception
        base_model = Xception(include_top=False, weights='imagenet',
                      input_shape=(PICS_WIDTH, PICS_HEIGHT, 3), pooling = 'avg')
    elif model_name == 'InceptionV3':
        from keras.applications.inception_v3 import InceptionV3
        base_model = InceptionV3(include_top=False, weights='imagenet',
                      input_shape=(PICS_WIDTH, PICS_HEIGHT, 3), pooling = 'avg')
    elif model_name == 'InceptionResNetV2':
        from keras.applications.inception_resnet_v2 import InceptionResNetV2
        base_model = InceptionResNetV2(include_top=False, weights='imagenet',
                      input_shape=(PICS_WIDTH, PICS_HEIGHT, 3), pooling = 'avg')
    elif model_name == 'NASNetLarge':
        from keras.applications.nasnet import NASNetLarge
        base_model = NASNetLarge(include_top=False, weights='imagenet',
                      input_shape=(PICS_WIDTH, PICS_HEIGHT, 3), pooling = 'avg')
    elif model_name == 'NASNetMobile':
        from keras.applications.nasnet import NASNetMobile
        base_model = NASNetMobile(include_top=False, weights='imagenet',
                      input_shape=(PICS_WIDTH, PICS_HEIGHT, 3), pooling = 'avg')
    else:
        raise ValueError('Network name is undefined')
        
    if head_only:
        for lay in base_model.layers:
            lay.trainable = False


    #manipulated = Input(shape=(1,))
    x = base_model.output
    #x = concatenate([x, manipulated])
    x = Dense(NUM_CATEGS, activation='softmax', name='predictions')(x)
    #model = Model([base_model.input,manipulated], x)
    model = Model(base_model.input, x)
    
    #print(model.summary())
    if weights != '':
        model.load_weights(weights)
    
    MODEL_OPTIMIZER = optimizers.SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss=MODEL_LOSS, optimizer=MODEL_OPTIMIZER, metrics=[MODEL_METRIC])

    return model

def get_preprocess(model_name):
    if model_name == 'VGG19':
        from keras.applications.vgg19 import preprocess_input
        return preprocess_input
    elif model_name == 'VGG16':
        from keras.applications.vgg16 import preprocess_input
        return preprocess_input
    elif model_name == 'MobileNet':
        from keras.applications.mobilenet import preprocess_input
        return preprocess_input
    elif model_name == 'ResNet50':
        from keras.applications.resnet50 import preprocess_input
        return preprocess_input
    elif model_name == 'Xception':
        from keras.applications.xception import preprocess_input
        return preprocess_input
    elif model_name == 'InceptionV3':
        from keras.applications.inception_v3 import preprocess_input
        return preprocess_input
    elif model_name == 'InceptionResNetV2':
        from keras.applications.inception_resnet_v2 import preprocess_input
        return preprocess_input
    elif model_name == 'NASNetLarge':
        from keras.applications.nasnet import preprocess_input
        return preprocess_input
    elif model_name == 'NASNetMobile':
        from keras.applications.nasnet import preprocess_input
        return preprocess_input
    else:
        raise ValueError('Network name is undefined')
        

def get_batch_size(model_name):
    if model_name == 'VGG19':
        return 24
    elif model_name == 'VGG16':
        return 24
    elif model_name == 'MobileNet':
        return 32
    elif model_name == 'ResNet50':
        return 16
    elif model_name == 'Xception':
        return 12
    elif model_name == 'InceptionV3':
        return 16
    elif model_name == 'InceptionResNetV2':
        return 12
    elif model_name == 'NASNetLarge':
        return 16
    elif model_name == 'NASNetMobile':
        return 16
    else:
        raise ValueError('Network name is undefined')
        

def make_cv_split(df_train,cameras,id_col,n_splits,path,seed):
    for camera in cameras:
        df = df_train[df_train.camera == camera].copy().reset_index(drop=True)
    
        skf = KFold(n_splits=n_splits,shuffle=True,random_state=seed)
        i=0
        for train_index, test_index in skf.split(df):
            id_train, id_test = df.loc[train_index,id_col], df.loc[test_index,id_col]
            id_train.to_csv(os.path.join(path,'CV_{}_{}_train.csv'.format(camera,i)),index=False,header=True)
            id_test.to_csv(os.path.join(path,'CV_{}_{}_test.csv'.format(camera,i)),index=False,header=True)
            i+=1
    
    for split in range(n_splits):
        id_list_train = []
        id_list_test = []
        for camera in cameras:
            id_list_train.append(pd.read_csv(os.path.join(path,'CV_{}_{}_train.csv'.format(camera,split))))
            id_list_test.append(pd.read_csv(os.path.join(path,'CV_{}_{}_test.csv'.format(camera,split))))
            os.remove(os.path.join(path,'CV_{}_{}_train.csv'.format(camera,split)))
            os.remove(os.path.join(path,'CV_{}_{}_test.csv'.format(camera,split)))
        id_train = pd.concat(id_list_train)
        id_test = pd.concat(id_list_test)
        
        id_train.to_csv(os.path.join(path,'CV_{}_train.csv'.format(split)),index=False,header=True)
        id_test.to_csv(os.path.join(path,'CV_{}_test.csv'.format(split)),index=False,header=True)
        
class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self): # Py3
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


# def train_batch_generator(paths, labels, batch_size, preprocess_input):
#     num_images = len(paths)
#     while True:
#         current_position = 0
#         while current_position < num_images:
#             if current_position + batch_size < num_images:
#                 idx_batch = np.arange(current_position,current_position + batch_size)
#             else:
#                 idx_batch = np.arange(current_position,num_images)    

#             X = pool.imap(read_image_train, paths[idx_batch])

#             X = np.array([preprocess_input(x.astype('float32')) for x in X]).astype('float32')
#             y = to_categorical(labels[idx_batch], num_classes=10)

#             current_position += batch_size
#             yield X, y

            
def train_batch_generator(paths, labels, batch_size, preprocess_input):
    num_images = len(paths)
    while True:
        idx_batch = np.random.randint(low=0, high=num_images, size=batch_size)
        O = pool.imap(read_image_train, paths[idx_batch])
        
        X_init = [x for x in O]
        
        X = np.array([preprocess_input(x[0].astype('float32')) for x in X_init]).astype('float32')
        M = np.array([x[1] for x in X_init])
        y = to_categorical(labels[idx_batch], num_classes=10)
        
        #yield [X,M], y
        yield X, y


def evaluation_batch_generator(paths, labels, batch_size, preprocess_input):
    num_images = len(paths)
    while True:
        current_position = 0
        while current_position < num_images:
            if current_position + batch_size < num_images:
                idx_batch = np.arange(current_position,current_position + batch_size)
            else:
                idx_batch = np.arange(current_position,num_images)    

            O = pool.imap(read_image_valid, paths[idx_batch])
            
            X_init = [x for x in O]
            
            X = np.array([preprocess_input(x[0].astype('float32')) for x in X_init]).astype('float32')
            M = np.array([x[1] for x in X_init])
            y = to_categorical(labels[idx_batch], num_classes=10)

            current_position += batch_size

            #yield [X,M], y
            yield X, y


def read_image_train(path):

    img = cv2.imread(path,cv2.IMREAD_COLOR)
    is_manip = 0.
    
    # Augmentation
    if AUGMENTATION == 'Train':
        # Resizing
        if random.randint(0,1):
            factor = random.randint(45,210)/100.
            min_factor = factor
            while (int(img.shape[0]*factor) < PICS_HEIGHT)|(int(img.shape[1]*factor) < PICS_WIDTH):
                factor = random.randint(int(min_factor*100+2),210)/100.
                mic_factor = factor
            img = cv2.resize(img, (int(img.shape[1]*factor), int(img.shape[0]*factor)), interpolation=cv2.INTER_CUBIC)
        # Gamma
        if random.randint(0,1):
            gamma = random.randint(70,130)/100.
            img = adjust_gamma(img, gamma=gamma)
        # Quality
        if random.randint(0,1):
            qual = random.randint(60,100)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(qual)]
            _, img = cv2.imencode('.jpg', img, encode_param)
            img = cv2.imdecode(img, 1)

        # Mirroring
        #if random.randint(0, 1):
        #    img = img[::-1, :, :]
        angle = random.randint(0, 3)
        if angle != 0:
            img = np.rot90(img, k=angle)
        
        # Random Crops
        border_height = 0
        border_width = 0
        sq_height = random.randint(0, img.shape[0] - PICS_HEIGHT - border_height)
        sq_width = random.randint(0, img.shape[1] - PICS_WIDTH - border_width)
        img = img[border_height + sq_height:border_height + sq_height + PICS_HEIGHT,
                  border_width + sq_width:border_width + sq_width + PICS_WIDTH,:]
        
        return img[:,:,::-1]
        #return img
     
    elif AUGMENTATION == 'PseudoLabeling':
        
        if path.split('/')[0] != 'test':
            
            if random.randint(0,1):
                is_manip = 1.
                which_aug = random.randint(0,2)
                if which_aug == 0:
                    factors = [0.5,0.8,1.5,2.0]
                    factor = random.randint(0,len(factors)-1)
                    min_factor = factor
                    while (int(img.shape[0]*factors[factor]) < PICS_HEIGHT)|(int(img.shape[1]*factors[factor]) < PICS_WIDTH):
                        min_factor += 1
                        factor = random.randint(min_factor,len(factors)-1)
                    aug = "resize_{}".format(factors[factor])
                    img = cv2.resize(img, (int(img.shape[1]*factors[factor]), int(img.shape[0]*factors[factor])), interpolation=cv2.INTER_CUBIC)
                elif which_aug == 1:
                    gammas = [0.8, 1.2]
                    gamma = random.randint(0,len(gammas)-1)
                    img = adjust_gamma(img, gamma=gammas[gamma])
                    aug = "gamma_{}".format(gammas[gamma])
                elif which_aug == 2:
                    quals = [70,90]
                    qual = random.randint(0,len(quals)-1)
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(quals[qual])]
                    _, img = cv2.imencode('.jpg', img, encode_param)
                    img = cv2.imdecode(img, 1)
                    aug = "jpg_{}".format(int(quals[qual]))

            # Mirroring
            if random.randint(0, 1):
                img = img[::-1, :, :]
            angle = random.randint(0, 3)
            if angle != 0:
                img = np.rot90(img, k=angle)

            # Random Crops
            border_height = 0
            border_width = 0
            sq_height = random.randint(0, img.shape[0] - PICS_HEIGHT - border_height)
            sq_width = random.randint(0, img.shape[1] - PICS_WIDTH - border_width)
            img = img[border_height + sq_height:border_height + sq_height + PICS_HEIGHT,
                      border_width + sq_width:border_width + sq_width + PICS_WIDTH,:]
            
            
            return [img[:,:,::-1], is_manip]
            
        else:
            if 'manip' in path:
                is_manip = 1.
            # Augmentation
            # Mirroring
            if random.randint(0, 1):
                img = img[::-1, :, :]
            angle = random.randint(0, 3)
            if angle != 0:
                img = np.rot90(img, k=angle)

            # *********************
            # Random Crops
            border_height = 0
            border_width = 0
            sq_height = random.randint(0, img.shape[0] - PICS_HEIGHT - border_height)
            sq_width = random.randint(0, img.shape[1] - PICS_WIDTH - border_width)
            img = img[border_height + sq_height:border_height + sq_height + PICS_HEIGHT,
                      border_width + sq_width:border_width + sq_width + PICS_WIDTH,:]

            #return img
            return [img[:,:,::-1], is_manip]
    
    if random.randint(0, 1):
        img = img[::-1, :, :]
    angle = random.randint(0, 3)
    if angle != 0:
        img = np.rot90(img, k=angle)
    
    # Random Crops
    border_height = 0
    border_width = 0
    sq_height = random.randint(0, img.shape[0] - PICS_HEIGHT - border_height)
    sq_width = random.randint(0, img.shape[1] - PICS_WIDTH - border_width)
    img = img[border_height + sq_height:border_height + sq_height + PICS_HEIGHT,
              border_width + sq_width:border_width + sq_width + PICS_WIDTH,:]
    
    return [img[:,:,::-1], is_manip]
    

def read_image_valid(path):
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    is_manip = 1. if 'manip' in path else 0.

    border_height = int((img.shape[0]-PICS_HEIGHT)/2)
    border_width = int((img.shape[1]-PICS_WIDTH)/2)

    img = img[border_height:border_height + PICS_HEIGHT,
              border_width:border_width + PICS_WIDTH,:]
    return img[:,:,::-1], is_manip

def read_image_test(path, preprocess_input):
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    is_manip = 1. if 'manip' in path else 0.
    
    border_height = int((img.shape[0]-PICS_HEIGHT)/2)
    border_width = int((img.shape[1]-PICS_WIDTH)/2)

    img = img[border_height:border_height + PICS_HEIGHT,
              border_width:border_width + PICS_WIDTH,:]

    return preprocess_input(img[:,:,::-1].astype('float32')), is_manip

def read_image_test_tta_5(path, preprocess_input):
    img = cv2.imread(path,cv2.IMREAD_COLOR)

    img_list = []
    
    # Crop Positions
    img_list.append(preprocess_input(img[:PICS_HEIGHT,:PICS_HEIGHT,::-1].astype('float32')))
    img_list.append(preprocess_input(img[-PICS_HEIGHT:,-PICS_HEIGHT:,::-1].astype('float32')))
    img_list.append(preprocess_input(img[:PICS_HEIGHT,-PICS_HEIGHT:,::-1].astype('float32')))
    img_list.append(preprocess_input(img[-PICS_HEIGHT:,:PICS_HEIGHT,::-1].astype('float32')))
    
    # Center Crop
    border_height = int((img.shape[0]-PICS_HEIGHT)/2)
    border_width = int((img.shape[1]-PICS_WIDTH)/2)
    img = img[border_height:border_height + PICS_HEIGHT, border_width:border_width + PICS_WIDTH,::-1]
    
    img_list.append(preprocess_input(img.astype('float32')))
    return img_list

def read_image_test_tta_9(path, preprocess_input):
    img = cv2.imread(path,cv2.IMREAD_COLOR)
    is_manip = 1. if 'manip' in path else 0.

    img_list = []
    
    # Crop Positions
    img_list.append(preprocess_input(img[:PICS_HEIGHT,:PICS_HEIGHT,::-1].astype('float32')))
    img_list.append(preprocess_input(img[-PICS_HEIGHT:,-PICS_HEIGHT:,::-1].astype('float32')))
    img_list.append(preprocess_input(img[:PICS_HEIGHT,-PICS_HEIGHT:,::-1].astype('float32')))
    img_list.append(preprocess_input(img[-PICS_HEIGHT:,:PICS_HEIGHT,::-1].astype('float32')))
    
    border_height = int((img.shape[0]-PICS_HEIGHT)/2)
    border_width = int((img.shape[1]-PICS_WIDTH)/2)
    
    img_list.append(preprocess_input(img[border_height:border_height + PICS_HEIGHT, border_width:border_width + PICS_WIDTH,::-1].astype('float32')))
    img_list.append(preprocess_input(img[:PICS_HEIGHT, border_width:border_width + PICS_WIDTH,::-1].astype('float32')))
    img_list.append(preprocess_input(img[-PICS_HEIGHT:, border_width:border_width + PICS_WIDTH,::-1].astype('float32')))
    img_list.append(preprocess_input(img[border_height:border_height + PICS_HEIGHT, :PICS_WIDTH,::-1].astype('float32')))
    img_list.append(preprocess_input(img[border_height:border_height + PICS_HEIGHT, -PICS_WIDTH:,::-1].astype('float32')))
    return img_list, [is_manip]*9

def read_image_test_tta_4(path, preprocess_input):
    img = cv2.imread(path,cv2.IMREAD_COLOR)

    img_list = []

    # Center Crop
    border_height = int((img.shape[0]-PICS_HEIGHT)/2)
    border_width = int((img.shape[1]-PICS_WIDTH)/2)
    img = img[border_height:border_height + PICS_HEIGHT,
              border_width:border_width + PICS_WIDTH,::-1]

    # Rotations
    for k in range(3):
        img_list.append(preprocess_input(np.rot90(img, k=k+1).astype('float32')))
    img_list.append(preprocess_input(img.astype('float32')))

    return img_list

def read_image_test_tta_20(path, preprocess_input):
    img = cv2.imread(path,cv2.IMREAD_COLOR)

    img_list = []
    
    border_height = int((img.shape[0]-PICS_HEIGHT)/2)
    border_width = int((img.shape[1]-PICS_WIDTH)/2)
    
    # Crop Positions and Rotations:
    for k in range(4):
        img_list.append(preprocess_input(np.rot90(img[:PICS_HEIGHT,:PICS_HEIGHT,::-1], k=k).astype('float32')))
        img_list.append(preprocess_input(np.rot90(img[-PICS_HEIGHT:,-PICS_HEIGHT:,::-1], k=k).astype('float32')))
        img_list.append(preprocess_input(np.rot90(img[:PICS_HEIGHT,-PICS_HEIGHT:,::-1], k=k).astype('float32')))
        img_list.append(preprocess_input(np.rot90(img[-PICS_HEIGHT:,:PICS_HEIGHT,::-1], k=k).astype('float32')))
        img_list.append(preprocess_input(np.rot90(img[border_height:border_height + PICS_HEIGHT, border_width:border_width + PICS_WIDTH,::-1], k=k).astype('float32')))
    
    return img_list

def read_image_test_tta_36(path, preprocess_input):
    img = cv2.imread(path,cv2.IMREAD_COLOR)

    img_list = []
    
    border_height = int((img.shape[0]-PICS_HEIGHT)/2)
    border_width = int((img.shape[1]-PICS_WIDTH)/2)
    
    # Crop Positions and Rotations:
    for k in range(4):
        img_list.append(preprocess_input(np.rot90(img[:PICS_HEIGHT,:PICS_HEIGHT,::-1], k=k).astype('float32')))
        img_list.append(preprocess_input(np.rot90(img[-PICS_HEIGHT:,-PICS_HEIGHT:,::-1], k=k).astype('float32')))
        img_list.append(preprocess_input(np.rot90(img[:PICS_HEIGHT,-PICS_HEIGHT:,::-1], k=k).astype('float32')))
        img_list.append(preprocess_input(np.rot90(img[-PICS_HEIGHT:,:PICS_HEIGHT,::-1], k=k).astype('float32')))
        img_list.append(preprocess_input(np.rot90(img[border_height:border_height + PICS_HEIGHT, border_width:border_width + PICS_WIDTH,::-1], k=k).astype('float32')))
        
        img_list.append(preprocess_input(np.rot90(img[:PICS_HEIGHT, border_width:border_width + PICS_WIDTH,::-1], k=k).astype('float32')))
        img_list.append(preprocess_input(np.rot90(img[-PICS_HEIGHT:, border_width:border_width + PICS_WIDTH,::-1], k=k).astype('float32')))
        img_list.append(preprocess_input(np.rot90(img[border_height:border_height + PICS_HEIGHT, :PICS_WIDTH,::-1], k=k).astype('float32')))
        img_list.append(preprocess_input(np.rot90(img[border_height:border_height + PICS_HEIGHT, -PICS_WIDTH:,::-1], k=k).astype('float32')))

    return img_list

def read_image_test_tta_72(path, preprocess_input):
    img = cv2.imread(path,cv2.IMREAD_COLOR)

    img_list = []
    
    border_height = int((img.shape[0]-PICS_HEIGHT)/2)
    border_width = int((img.shape[1]-PICS_WIDTH)/2)
    
    # Crop Positions and Rotations:
    for k in range(4):
        img_list.append(preprocess_input(np.rot90(img[:PICS_HEIGHT,:PICS_HEIGHT,::-1], k=k).astype('float32')))
        img_list.append(preprocess_input(np.rot90(img[-PICS_HEIGHT:,-PICS_HEIGHT:,::-1], k=k).astype('float32')))
        img_list.append(preprocess_input(np.rot90(img[:PICS_HEIGHT,-PICS_HEIGHT:,::-1], k=k).astype('float32')))
        img_list.append(preprocess_input(np.rot90(img[-PICS_HEIGHT:,:PICS_HEIGHT,::-1], k=k).astype('float32')))
        img_list.append(preprocess_input(np.rot90(img[border_height:border_height + PICS_HEIGHT, border_width:border_width + PICS_WIDTH,::-1], k=k).astype('float32')))
        
        img_list.append(preprocess_input(np.rot90(img[:PICS_HEIGHT, border_width:border_width + PICS_WIDTH,::-1], k=k).astype('float32')))
        img_list.append(preprocess_input(np.rot90(img[-PICS_HEIGHT:, border_width:border_width + PICS_WIDTH,::-1], k=k).astype('float32')))
        img_list.append(preprocess_input(np.rot90(img[border_height:border_height + PICS_HEIGHT, :PICS_WIDTH,::-1], k=k).astype('float32')))
        img_list.append(preprocess_input(np.rot90(img[border_height:border_height + PICS_HEIGHT, -PICS_WIDTH:,::-1], k=k).astype('float32')))
        
        img_list.append(preprocess_input(np.flip(np.rot90(img[:PICS_HEIGHT,:PICS_HEIGHT,::-1], k=k),0).astype('float32')))
        img_list.append(preprocess_input(np.flip(np.rot90(img[-PICS_HEIGHT:,-PICS_HEIGHT:,::-1], k=k),0).astype('float32')))
        img_list.append(preprocess_input(np.flip(np.rot90(img[:PICS_HEIGHT,-PICS_HEIGHT:,::-1], k=k),0).astype('float32')))
        img_list.append(preprocess_input(np.flip(np.rot90(img[-PICS_HEIGHT:,:PICS_HEIGHT,::-1], k=k),0).astype('float32')))
        img_list.append(preprocess_input(np.flip(np.rot90(img[border_height:border_height + PICS_HEIGHT, border_width:border_width + PICS_WIDTH,::-1], k=k),0).astype('float32')))
        
        img_list.append(preprocess_input(np.flip(np.rot90(img[:PICS_HEIGHT, border_width:border_width + PICS_WIDTH,::-1], k=k),0).astype('float32')))
        img_list.append(preprocess_input(np.flip(np.rot90(img[-PICS_HEIGHT:, border_width:border_width + PICS_WIDTH,::-1], k=k),0).astype('float32')))
        img_list.append(preprocess_input(np.flip(np.rot90(img[border_height:border_height + PICS_HEIGHT, :PICS_WIDTH,::-1], k=k),0).astype('float32')))
        img_list.append(preprocess_input(np.flip(np.rot90(img[border_height:border_height + PICS_HEIGHT, -PICS_WIDTH:,::-1], k=k),0).astype('float32')))

    return img_list
   
    
def model_predict(test_fnames,model_name,weights_path,TTA=False,augment_number_per_image=1):
    preprocessing = get_preprocess(model_name)
    TRAIN_BATCH_SIZE = get_batch_size(model_name)
    EVALUATION_BATCH_SIZE = int(TRAIN_BATCH_SIZE*1.5)
    model = InitialiazeModel(head_only=False,weights=weights_path,model_name = model_name, lr=0.001)

    test_shape = test_fnames.shape[0]
    batch_len = EVALUATION_BATCH_SIZE * 32 // augment_number_per_image
    
    predictions_list = []
    current_position = 0
    while current_position < test_shape:
        if current_position + batch_len < test_shape:
            part_files = test_fnames[current_position:current_position + batch_len]
        else:
            part_files = test_fnames[current_position:]     

        img_list = []
        manip_list = []
        for fname in part_files:
            if TTA:
                # TTA -----------------------
                imgs = read_image_test_tta_36(fname,preprocessing)
                img_list.extend(imgs)
                #manip_list.extend(is_manips)
            else: 
                img, is_manip = read_image_test(fname,preprocessing)
                img_list.append(img)
                manip_list.append(is_manip)
        predictions_list.append(model.predict(np.array(img_list), batch_size=EVALUATION_BATCH_SIZE, verbose=1))

        current_position += batch_len
        print('Test percentage: ', str(current_position/test_shape))

    predictions = np.concatenate(predictions_list)
    
    # TTA ----------------------------
    if TTA:
        ans = []
        total = 0
        for _ in range(test_shape):
            part = []
            for _ in range(augment_number_per_image):
                part.append(predictions[total])
                total += 1
            part = np.mean(np.array(part), axis=0)
            ans.append(part)
        predictions = np.array(ans)
 
    preds_camera = np.argmax(predictions,axis=1)
    return (predictions,preds_camera) 