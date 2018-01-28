from functools import partial

from keras.applications.mobilenet import MobileNet, preprocess_input
# from keras.applications.xception import Xception, preprocess_input

from keras.models import Model
from keras.layers import Input, Lambda, Dense, Conv2D, BatchNormalization, GlobalAveragePooling2D, Dropout
from keras import backend as K

from src.aug import augment
from src.dataset import PseudoDataset, SiameseWrapper
from src.model import get_callbacks


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))


def make_siamese_model(shape):
    shape = shape, shape, 3
    # feature_extractor = MobileNet(input_shape=shape, include_top=False, pooling='avg')
    feature_extractor = make_dummynet(shape)

    input_a = Input(shape=shape)
    input_b = Input(shape=shape)

    processed_a = feature_extractor(input_a)
    processed_b = feature_extractor(input_b)

    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    distance = Dense(1, activation='sigmoid')(distance)

    model = Model([input_a, input_b], distance)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def make_dummynet(shape):
    p = .5

    inp = Input(shape)
    x = Conv2D(8, kernel_size=3, padding='valid', activation='relu')(inp)
    x = Dropout(p)(x)
    x = Conv2D(16, kernel_size=3, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, kernel_size=3, padding='valid', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, kernel_size=3, padding='valid', activation='relu')(x)
    out = GlobalAveragePooling2D()(x)
    model = Model(inputs=inp, outputs=out)

    return model


def main(n_fold=0, batch_size=64):
    shape = 224
    aug = partial(augment, expected_shape=shape)

    n_fold = int(n_fold)
    batch_size = int(batch_size)
    model = make_siamese_model(shape)

    train = PseudoDataset(n_fold=n_fold,
                          batch_size=batch_size,
                          transform=preprocess_input,
                          train=True,
                          size=shape,
                          aug=aug,
                          center_crop_size=0,
                          )

    val = PseudoDataset(n_fold=n_fold,
                        transform=preprocess_input,
                        batch_size=batch_size,
                        train=False,
                        size=shape,
                        aug=aug,
                        center_crop_size=0)

    model.fit_generator(SiameseWrapper(train),
                        epochs=500,
                        steps_per_epoch=50,
                        validation_data=SiameseWrapper(val),
                        workers=8,
                        max_queue_size=32,
                        use_multiprocessing=False,
                        validation_steps=10,
                        callbacks=get_callbacks('siamese', loss_name='euclidian', dataset_name='pseudo', fold=n_fold),
                        )


if __name__ == '__main__':
    main()
