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
    tensorboard = TensorBoard(log_dir="./logs", histogram_freq=0,
                              batch_size=32, write_graph=True,
                              write_grads=True)
    return [terminator, checkpointer, tensorboard]


def generator(image_list, images_path, expected_images, size=1):
    while True:
        batch_names = numpy.random.choice(image_list, size=size)
        batch_img = []
        batch_expected = []
        batch_slic = []
        batch_vertices = []
        batch_neighbors = []
        for image_name in batch_names:
            # LOAD IMAGES
            img = resize(io.imread(images_path + image_name + ".jpg"), IMAGE_SHAPE, anti_aliasing=True)
            expected = resize(io.imread(expected_images + image_name + ".png"), IMAGE_SHAPE, anti_aliasing=True)

            # OBTAIN OTHER USEFUL DATA
            slic = obtain_superpixels(img, N_SUPERPIXELS, SLIC_SIGMA)
            vertices = average_rgb_for_superpixels(img, slic)
            neighbors = get_neighbors(slic, N_SUPERPIXELS)
            expected = average_rgb_for_superpixels(expected, slic)

            # ADD TO BATCH
            batch_img += [img]
            batch_expected += [expected]
            batch_slic += [slic]
            batch_vertices += [vertices]
            batch_neighbors += [neighbors]
        batch_img = numpy.array(batch_img)
        batch_expected = numpy.array(batch_expected)
        batch_slic = numpy.array(batch_slic)
        batch_vertices = numpy.array(batch_vertices)
        batch_neighbors = numpy.array(batch_neighbors)
        yield ([batch_img, batch_slic, batch_vertices, batch_neighbors], [batch_expected])


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
    model.fit_generator(generator(train_image_list, IMAGES_PATH, VALIDATION_IMAGES, TRAIN_BATCH_SIZE),
                        steps_per_epoch=numpy.ceil(
                            TRAIN_ELEMS / (TRAIN_BATCH_SIZE*48)),
                        epochs=7,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=generator(val_image_list, IMAGES_PATH, VALIDATION_IMAGES, VALIDATION_BATCH_SIZE),
                        validation_steps=numpy.ceil(
                            VALIDATION_ELEMS / (VALIDATION_BATCH_SIZE*48)),
                        max_queue_size=10,
                        shuffle=True)
    model.save(MODEL_PATH)
