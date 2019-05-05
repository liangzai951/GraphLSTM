import numpy
from keras.callbacks import TerminateOnNaN, ModelCheckpoint, TensorBoard
from keras.engine.saving import load_model
from skimage import io
from skimage.transform import resize

from config import *
from layers.ConfidenceLayer import Confidence
from layers.GraphLSTM import GraphLSTM
from layers.GraphLSTMCell import GraphLSTMCell
from layers.GraphPropagation import GraphPropagation
from layers.InverseGraphPropagation import InverseGraphPropagation
from utils.utils import obtain_superpixels, get_neighbors, \
    average_rgb_for_superpixels


def init_callbacks():
    terminator = TerminateOnNaN()
    checkpointer = ModelCheckpoint(
        "./checkpoints/model_{epoch:02d}_{val_acc:.2f}.hdf5",
        monitor="val_acc", save_weights_only=False, mode="max", period=2)
    return [terminator, checkpointer]


def generator(image_list, images_path, expected_images, size=1):
    while True:
        batch_names = numpy.random.choice(image_list, size=size)
        batch_expected = []
        batch_vertices = []
        batch_neighbors = []
        for image_name in batch_names:
            # LOAD IMAGES
            img = resize(io.imread(images_path + image_name + ".png"), IMAGE_SHAPE, anti_aliasing=True)
            expected = resize(io.imread(images_path + image_name + ".png"), IMAGE_SHAPE, anti_aliasing=True)

            # OBTAIN OTHER USEFUL DATA
            slic = obtain_superpixels(img, N_SUPERPIXELS, SLIC_SIGMA)
            vertices = average_rgb_for_superpixels(img, slic)
            neighbors = get_neighbors(slic, N_SUPERPIXELS)
            expected = average_rgb_for_superpixels(expected, slic)

            # ADD TO BATCH
            batch_expected += [expected]
            batch_vertices += [vertices]
            batch_neighbors += [neighbors]
        batch_expected = numpy.array(batch_expected)
        batch_vertices = numpy.array(batch_vertices)
        batch_neighbors = numpy.array(batch_neighbors)
        yield ([batch_vertices, batch_neighbors], [batch_expected])


if __name__ == '__main__':
    callbacks = init_callbacks()

    train_image_list = ["test_{0!s}".format(i) for i in range(20)]
    val_image_list = train_image_list[:4]
    train_image_list = train_image_list[4:]

    model = load_model("glstm_test.hdf5",
                       custom_objects={'Confidence': Confidence,
                                       'GraphPropagation': GraphPropagation,
                                       'InverseGraphPropagation': InverseGraphPropagation,
                                       'GraphLSTM': GraphLSTM,
                                       'GraphLSTMCell': GraphLSTMCell,
                                       'custom_mse': custom_mse})
    # model = create_model()
    model.fit_generator(generator(train_image_list, "../../data/test/", "../../data/test/", TRAIN_BATCH_SIZE),
                        steps_per_epoch=numpy.ceil(
                            16 / (TRAIN_BATCH_SIZE)),
                        epochs=50,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=generator(val_image_list, "../../data/test/", "../../data/test/", VALIDATION_BATCH_SIZE),
                        validation_steps=numpy.ceil(
                            4 / (VALIDATION_BATCH_SIZE)),
                        max_queue_size=10,
                        shuffle=True)
    model.save(MODEL_PATH)
