from functools import partial

from keras.optimizers import SGD
from fire import Fire

from src.dataset import get_dataset
from src.model import get_model, get_callbacks
from src.aug import augment
from src.utils import logger


def fit_once(model, model_name, loss, train, val, dataset_name, n_fold, start_epoch):
    steps_per_epoch = 500
    validation_steps = 100

    model.compile(optimizer=SGD(clipvalue=4, momentum=.9, nesterov=True), loss=loss,
                  metrics=['accuracy'])
    history = model.fit_generator(train,
                                  epochs=500,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=val,
                                  workers=8,
                                  max_queue_size=32,
                                  use_multiprocessing=False,
                                  validation_steps=validation_steps,
                                  callbacks=get_callbacks(model_name, loss, dataset_name, n_fold),
                                  initial_epoch=start_epoch,
                                  )
    return model, max(history.epoch)


def fit_model(dataset_name, model_name, batch_size=16, n_fold=1, shape=384):
    dataset, n_classes = get_dataset(dataset_name)
    aug = partial(augment, expected_shape=shape)

    n_fold = int(n_fold)
    batch_size = int(batch_size)
    model, preprocess = get_model(model_name, shape, n_classes=n_classes)

    train = dataset(n_fold=n_fold,
                    batch_size=batch_size,
                    transform=preprocess,
                    train=True,
                    size=shape,
                    aug=aug,
                    center_crop_size=0)

    val = dataset(n_fold=n_fold,
                  transform=preprocess,
                  batch_size=batch_size,
                  train=False,
                  size=shape,
                  aug=aug,
                  center_crop_size=0)

    frozen_epochs = 1
    steps_per_epoch = 500
    validation_steps = 50
    loss = 'categorical_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    model.fit_generator(train,
                        epochs=frozen_epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val,
                        workers=8,
                        validation_steps=validation_steps,
                        use_multiprocessing=False,
                        callbacks=get_callbacks(model_name, loss, dataset_name, n_fold),
                        max_queue_size=50,
                        )

    for layer in model.layers:
        layer.trainable = True

    model, epoch = fit_once(model=model,
                            model_name=model_name,
                            loss='categorical_crossentropy',
                            train=train,
                            val=val,
                            start_epoch=frozen_epochs,
                            dataset_name=dataset_name,
                            n_fold=n_fold,
                            )

    model, epoch = fit_once(model=model,
                            model_name=model_name,
                            loss='categorical_hinge',
                            train=train,
                            val=val,
                            start_epoch=epoch,
                            dataset_name=dataset_name,
                            n_fold=n_fold,
                            )

    train = dataset(n_fold=n_fold,
                    batch_size=batch_size,
                    transform=preprocess,
                    train=True,
                    size=shape,
                    aug=aug,
                    center_crop_size=1024)

    val = dataset(n_fold=n_fold,
                  transform=preprocess,
                  batch_size=batch_size,
                  train=False,
                  size=shape,
                  aug=aug,
                  center_crop_size=1024)

    logger.info('Fitting again with center crop')

    model, epoch = fit_once(model=model,
                            model_name=f'{model_name}_crop',
                            loss='categorical_crossentropy',
                            train=train,
                            val=val,
                            start_epoch=epoch,
                            dataset_name=dataset_name,
                            n_fold=n_fold,
                            )

    fit_once(model=model,
             model_name=f'{model_name}_crop',
             loss='categorical_hinge',
             train=train,
             val=val,
             start_epoch=epoch,
             dataset_name=dataset_name,
             n_fold=n_fold,
             )


if __name__ == '__main__':
    Fire(fit_model)
