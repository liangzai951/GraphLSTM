from random import shuffle

import numpy
from skimage import io

from keras.callbacks import TerminateOnNaN, ModelCheckpoint, TensorBoard
from keras.engine.saving import load_model
from skimage.transform import resize

from config import *
from create_model import create_model
from layers.GraphLSTM import GraphLSTM
from layers.GraphLSTMCell import GraphLSTMCell
from layers.GraphPropagation import GraphPropagation
from layers.ConfidenceLayer import Confidence
from layers.InverseGraphPropagation import InverseGraphPropagation
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


def generator(image_list, images_path, expected_images):
    while True:
        shuffle(image_list)
        for image_name in image_list:
            # LOAD IMAGES
            img = resize(io.imread(images_path + image_name + ".jpg"), IMAGE_SHAPE, anti_aliasing=True)
            expected = resize(io.imread(expected_images + image_name + ".png"), IMAGE_SHAPE, anti_aliasing=True)

            # OBTAIN OTHER USEFUL DATA
            slic = obtain_superpixels(img, N_SUPERPIXELS, SLIC_SIGMA)
            vertices = average_rgb_for_superpixels(img, slic)
            neighbors = get_neighbors(slic, N_SUPERPIXELS)
            expected = average_rgb_for_superpixels(expected, slic)

            # TO NUMPIES
            img = numpy.expand_dims(numpy.array(img), axis=0)
            expected = numpy.expand_dims(numpy.array(expected), axis=0)
            slic = numpy.expand_dims(numpy.array(slic), axis=0)
            vertices = numpy.expand_dims(numpy.array(vertices), axis=0)
            neighbors = numpy.expand_dims(numpy.array(neighbors), axis=0)

            yield ([img, slic, vertices, neighbors], [expected])


if __name__ == '__main__':
    callbacks = init_callbacks()

    with open(TRAINSET_FILE) as f:
        train_image_list = [line.replace("\n", "") for line in f]
    with open(TRAINVALSET_FILE) as f:
        val_image_list = [line.replace("\n", "") for line in f]

    model = load_model(MODEL_PATH,
                       custom_objects={'Confidence': Confidence,
                                       'GraphPropagation': GraphPropagation,
                                       'InverseGraphPropagation': InverseGraphPropagation})
    # model = create_model()
    model.fit_generator(generator(train_image_list, IMAGES_PATH, VALIDATION_IMAGES),
                        steps_per_epoch=numpy.ceil(
                            TRAIN_ELEMS / TRAIN_BATCH_SIZE),
                        epochs=100,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=generator(val_image_list, IMAGES_PATH, VALIDATION_IMAGES),
                        validation_steps=numpy.ceil(
                            VALIDATION_ELEMS / VALIDATION_BATCH_SIZE),
                        max_queue_size=10,
                        shuffle=True)
