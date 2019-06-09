import random
import matplotlib.pyplot as plt
import numpy
from keras.callbacks import TerminateOnNaN, ModelCheckpoint
from keras.engine.saving import load_model
from skimage import io
from skimage.transform import resize

from config import *
from layers.GraphLSTM import GraphLSTM
from layers.GraphLSTMCell import GraphLSTMCell
from utils.utils import obtain_superpixels, get_neighbors, \
    average_rgb_for_superpixels, sort_values, \
    get_superpixels_index_for_hot_areas


def init_callbacks(t):
    terminator = TerminateOnNaN()
    checkpointer = ModelCheckpoint(
        "../data/checkpoints/model"+str(t)+"_{epoch:02d}_{loss:.2f}_{val_loss:.2f}.hdf5",
        monitor="val_loss", save_weights_only=False, mode="min", period=5)
    return [terminator, checkpointer]


def generator(image_list, images_path, expected_images, size=1):
    while True:
        batch_names = numpy.random.choice(image_list, size=size)

        batch_vertices = []
        batch_neighbors = []

        batch_inputs = [[] for _ in range(INPUT_PATHS)]
        batch_indexes = [[] for _ in range(INPUT_PATHS)]
        batch_r_indexes = [[] for _ in range(INPUT_PATHS)]

        for image_name in batch_names:
            # LOAD IMAGES
            img = resize(io.imread(images_path + image_name + ".png"),
                         IMAGE_SHAPE, anti_aliasing=True)

            # OBTAIN OTHER USEFUL DATA
            slic = obtain_superpixels(img, N_SUPERPIXELS, SLIC_SIGMA)
            vertices = average_rgb_for_superpixels(img, slic)
            neighbors = get_neighbors(slic, N_SUPERPIXELS)
            areas = get_superpixels_index_for_hot_areas(slic)
            for paths in range(INPUT_PATHS):
                vertex_index = areas[paths]
                path, mapping, r_mapping = sort_values(vertices, neighbors, vertex_index, mode="bfs")
                batch_inputs[paths].append(path)
                batch_indexes[paths].append(mapping)
                batch_r_indexes[paths].append(r_mapping)

            # ADD TO BATCH
            batch_vertices += [vertices]
            batch_neighbors += [neighbors]

        batch_vertices = numpy.array(batch_vertices)
        batch_neighbors = numpy.array(batch_neighbors)
        batch_inputs = [numpy.array(i) for i in batch_inputs]
        batch_indexes = [numpy.array(i) for i in batch_indexes]
        batch_r_indexes = [numpy.array(i) for i in batch_r_indexes]

        yield ([batch_vertices, batch_neighbors] +
               batch_inputs +
               batch_indexes +
               batch_r_indexes,
               [batch_inputs[0]])


if __name__ == '__main__':
    for t in range(50):
        callbacks = init_callbacks(t)
        val_image_list = image_list[:VALIDATION_ELEMS]
        train_image_list = image_list[VALIDATION_ELEMS:]

        model = load_model("glstm_raw{0!s}.hdf5".format(t),
                           custom_objects={'GraphLSTM': GraphLSTM,
                                           'GraphLSTMCell': GraphLSTMCell})
        # model = create_model()
        history = model.fit_generator(
            generator(train_image_list, "../data/test/", "../data/test/",
                      TRAIN_BATCH_SIZE),
            steps_per_epoch=numpy.ceil(TRAIN_ELEMS / TRAIN_BATCH_SIZE),
            epochs=EPOCHS,
            verbose=1,
            callbacks=callbacks,
            validation_data=generator(val_image_list, "../data/test/",
                                      "../data/test/", VALIDATION_BATCH_SIZE),
            validation_steps=numpy.ceil(VALIDATION_ELEMS / VALIDATION_BATCH_SIZE),
            max_queue_size=10,
            shuffle=True)
        model.save("glstm{0!s}.hdf5".format(t))

        plt.plot(history.history['acc'], color="#FF3864")
        plt.plot(history.history['val_acc'], color="#261447")
        plt.title('Accuracy - model {0!s}'.format(t))
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig('model_accuracy_{0!s}.png'.format(t))
        # summarize history for loss
        plt.plot(history.history['loss'], color="#FF3864")
        plt.plot(history.history['val_loss'], color="#261447")
        plt.title('Loss - model {0!s}'.format(t))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig('model_loss_{0!s}.png'.format(t))
