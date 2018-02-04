from functools import partial

from keras.optimizers import SGD
from fire import Fire

from src.dataset import MixedDataset, ExtraDataset, DatasetCollection
from src.model import get_model, get_callbacks
from src.aug import augment
from src.utils import logger


def fit_once(model, model_name, loss, train, val, stage, n_fold, start_epoch, initial=False):
    logger.info(f'Stage {stage} started: loss {loss}, fold {n_fold}')
    steps_per_epoch = 500
    validation_steps = 100

    model.compile(optimizer=SGD(lr=0.001 if initial else 0.01, clipvalue=4, momentum=.9, nesterov=True),
                  loss=loss,
                  metrics=['accuracy'])
    history = model.fit_generator(train,
                                  epochs=500,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=val,
                                  workers=8,
                                  max_queue_size=32,
                                  use_multiprocessing=False,
                                  validation_steps=validation_steps,
                                  callbacks=get_callbacks(model_name, loss, stage, n_fold),
                                  initial_epoch=start_epoch,
                                  )
    return model, max(history.epoch)


def fit_model(model_name, batch_size=16, n_fold=1, shape=384):
    n_classes = 10
    aug = partial(augment, expected_shape=shape)

    n_fold = int(n_fold)
    batch_size = int(batch_size)
    model, preprocess = get_model(model_name, shape, n_classes=n_classes)

    def make_config(**kwargs):
        d = {'n_fold': n_fold,
             'transform': preprocess,
             'batch_size': batch_size,
             'train': True,
             'size': shape,
             'aug': aug,
             'center_crop_size': 0}
        d.update(kwargs)
        return d

    train_mixed = MixedDataset(**make_config())
    val_mixed = MixedDataset(**make_config(train=False))
    train_mixed_crop = MixedDataset(**make_config(center_crop_size=1024))
    val_mixed_crop = MixedDataset(**make_config(train=False, center_crop_size=1024))

    train_extra = ExtraDataset(**make_config())
    val_extra = ExtraDataset(**make_config(train=False))

    frozen_epochs = 1
    steps_per_epoch = 500
    validation_steps = 50
    loss = 'categorical_crossentropy'
    model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    model.fit_generator(DatasetCollection(train_mixed, train_extra),
                        epochs=frozen_epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=DatasetCollection(val_mixed, val_extra),
                        workers=8,
                        validation_steps=validation_steps,
                        use_multiprocessing=False,
                        max_queue_size=50,
                        )

    for layer in model.layers:
        layer.trainable = True

    epoch = frozen_epochs
    for stage, (train, val) in enumerate((
            (DatasetCollection(train_mixed, train_extra), DatasetCollection(val_mixed, val_extra)),
            (DatasetCollection(train_mixed), DatasetCollection(val_mixed)),
            (DatasetCollection(train_mixed_crop), DatasetCollection(val_mixed_crop)),
    )):
        model, epoch = fit_once(model=model,
                                model_name=model_name,
                                loss='categorical_crossentropy',
                                train=train,
                                val=val,
                                start_epoch=epoch,
                                stage=stage,
                                n_fold=n_fold,
                                initial=True if stage > 0 else False
                                )

        model, epoch = fit_once(model=model,
                                model_name=model_name,
                                loss='categorical_hinge',
                                train=train,
                                val=val,
                                start_epoch=epoch,
                                stage=stage,
                                n_fold=n_fold,
                                initial=True if stage > 0 else False
                                )


if __name__ == '__main__':
    Fire(fit_model)
