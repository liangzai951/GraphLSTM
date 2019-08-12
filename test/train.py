import copy
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
from layers.InverseGraphPropagation import InverseGraphPropagation
from utils.utils import obtain_superpixels, get_neighbors, \
    average_rgb_for_superpixels, sort_values, \
    get_superpixels_index_for_hot_areas


def init_callbacks(k):
    terminator = TerminateOnNaN()
    checkpointer = ModelCheckpoint(
        "../data/checkpoints/model"+str(k)+"_{epoch:02d}_{loss:.2f}_{val_loss:.2f}_{acc:.2f}_{val_acc:.2f}.hdf5",
        monitor="val_loss", save_weights_only=False, mode="min", period=5)
    return [terminator, checkpointer]


def generator(image_list, images_path, expected_images, k, size=1):
    while True:
        batch_names = numpy.random.choice(image_list, size=size)

        batch_vertices = []
        batch_neighbors = []
        batch_expected = []

        batch_inputs = [[] for _ in range(k)]
        batch_indexes = [[] for _ in range(k)]
        batch_r_indexes = [[] for _ in range(k)]

        for image_name in batch_names:
            # LOAD IMAGES
            img = resize(io.imread(images_path + image_name + ".png"),
                         IMAGE_SHAPE, anti_aliasing=True)

            # OBTAIN OTHER USEFUL DATA
            slic = obtain_superpixels(img, N_SUPERPIXELS, SLIC_SIGMA)
            vertices = average_rgb_for_superpixels(img, slic)
            neighbors = get_neighbors(slic, N_SUPERPIXELS)

            expected = copy.deepcopy(vertices)
            for i in expected:
                i.append(0.0)

            areas = get_superpixels_index_for_hot_areas(slic)
            for paths in range(k):
                vertex_index = random.choice(areas)
                path, mapping, r_mapping = sort_values(vertices, neighbors,
                                                       vertex_index,
                                                       mode="bfs")
                batch_inputs[paths].append(path)
                batch_indexes[paths].append(mapping)
                batch_r_indexes[paths].append(r_mapping)

            green_near_red_indices = []
            for i, v in enumerate(vertices):
                if v[0] != 1.0:
                    continue
                neighborhood_indexes = numpy.where(neighbors[i] == 1)[0]
                if any(vertices[n][1] == 1.0 for n in neighborhood_indexes):
                    green_near_red_indices.append(i)

            for i in green_near_red_indices:
                expected[i] = [0.0] * len(expected[i])
                expected[i][-1] = 1.0

            # ADD TO BATCH
            batch_vertices += [vertices]
            batch_neighbors += [neighbors]
            batch_expected += [expected]

        batch_vertices = numpy.array(batch_vertices)
        batch_neighbors = numpy.array(batch_neighbors)
        batch_expected = numpy.array(batch_expected)
        batch_inputs = [numpy.array(i) for i in batch_inputs]
        batch_indexes = [numpy.array(i) for i in batch_indexes]
        batch_r_indexes = [numpy.array(i) for i in batch_r_indexes]

        yield ([batch_vertices, batch_neighbors] +
               batch_inputs +
               batch_indexes +
               batch_r_indexes,
               [batch_expected])


if __name__ == '__main__':
    for k in [2, 3, 5, 7, 9]:
        callbacks = init_callbacks(k)
        val_image_list = image_list[:VALIDATION_ELEMS]
        train_image_list = image_list[VALIDATION_ELEMS:]

        model = load_model("glstm_raw{0!s}.hdf5".format(k),
                           custom_objects={'GraphLSTM': GraphLSTM,
                                           'GraphLSTMCell': GraphLSTMCell,
                                           'InverseGraphPropagation': InverseGraphPropagation})
        # model = create_model()
        history = model.fit_generator(
            generator(train_image_list, "../data/test/", "../data/test/", k,
                      TRAIN_BATCH_SIZE),
            steps_per_epoch=numpy.ceil(TRAIN_ELEMS / TRAIN_BATCH_SIZE),
            epochs=EPOCHS,
            verbose=1,
            callbacks=callbacks,
            validation_data=generator(val_image_list, "../data/test/",
                                      "../data/test/", k, VALIDATION_BATCH_SIZE),
            validation_steps=numpy.ceil(VALIDATION_ELEMS / VALIDATION_BATCH_SIZE),
            max_queue_size=10,
            shuffle=True)
        model.save("glstm{0!s}.hdf5".format(k))

        plt.plot(history.history['acc'], color="#FF3864")
        plt.plot(history.history['val_acc'], color="#261447")
        plt.title('Accuracy - {0!s} paths'.format(k))
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig('model_accuracy_{0!s}_paths.png'.format(k))
        # summarize history for loss
        plt.plot(history.history['loss'], color="#FF3864")
        plt.plot(history.history['val_loss'], color="#261447")
        plt.title('Loss - {0!s} paths'.format(k))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig('model_loss_{0!s}_paths.png'.format(k))
