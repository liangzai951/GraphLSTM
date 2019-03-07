from random import shuffle

from skimage import io

from keras.callbacks import TerminateOnNaN, ModelCheckpoint, TensorBoard
from keras.engine.saving import load_model

from config import *
from utils.utils import obtain_superpixels, get_neighbors, \
    average_rgb_for_superpixels


def init_callbacks():
    terminator = TerminateOnNaN()
    checkpointer = ModelCheckpoint(
        "./data/checkpoints/model_{epoch:02d}_{val_acc:.2f}.hdf5",
        monitor="val_acc",
        save_best_only=True, save_weights_only=False, mode="max", period=1)
    tensorboard = TensorBoard(log_dir="./logs", histogram_freq=1,
                              batch_size=32, write_graph=True,
                              write_grads=True)
    return [terminator, checkpointer, tensorboard]


def generator():
    with open(TRAINSET_FILE) as f:
        image_list = [line for line in f]
        shuffle(image_list)
    for image_name in image_list:
        img = io.imread(IMAGES_PATH + image_name + ".jpg")
        expected = io.imread(VALIDATION_IMAGES + image_name + ".png")
        slic = obtain_superpixels(img, N_SUPERPIXELS, SLIC_SIGMA)
        vertices = average_rgb_for_superpixels(img, slic)
        neighbors = get_neighbors(slic, N_SUPERPIXELS)
        expected = average_rgb_for_superpixels(expected, slic)
        yield ([img, slic, vertices, neighbors], [expected])


def validation_generator():
    with open(TRAINVALSET_FILE) as f:
        image_list = [line for line in f]
        shuffle(image_list)
    for image_name in image_list:
        img = io.imread(IMAGES_PATH + image_name + ".jpg")
        expected = io.imread(VALIDATION_IMAGES + image_name + ".png")
        slic = obtain_superpixels(img, N_SUPERPIXELS, SLIC_SIGMA)
        vertices = average_rgb_for_superpixels(img, slic)
        neighbors = get_neighbors(slic, N_SUPERPIXELS)
        expected = average_rgb_for_superpixels(expected, slic)
        yield ([img, slic, vertices, neighbors], [expected])


if __name__ == '__main__':
    callbacks = init_callbacks()
    model = load_model(MODEL_PATH)
    model.fit_generator(generator,
                        steps_per_epoch=None,
                        epochs=100,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=validation_generator,
                        max_queue_size=10,
                        workers=3,
                        use_multiprocessing=True,
                        shuffle=True)
