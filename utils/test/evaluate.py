import numpy
from keras.engine.saving import load_model
from skimage import io
from skimage.transform import resize
from tqdm import tqdm

from config import *
from layers.ConfidenceLayer import Confidence
from layers.GraphLSTM import GraphLSTM
from layers.GraphLSTMCell import GraphLSTMCell
from layers.GraphPropagation import GraphPropagation
from layers.InverseGraphPropagation import InverseGraphPropagation
from utils.utils import obtain_superpixels, average_rgb_for_superpixels, \
    get_neighbors

if __name__ == '__main__':
    OUTPUT_PATH = "./"
    image_list = ["test_{0!s}".format(i) for i in range(20)]

    while len(image_list) % PREDICT_BATCH_SIZE != 0:
        image_list.append(None)
    VALIDATION_IMAGES = IMAGES_PATH = "../../data/test/"
    MODEL_PATH = "./checkpoints/model_50_0.73.hdf5"
    model = load_model(MODEL_PATH, custom_objects={'Confidence': Confidence,
                                                   'GraphPropagation': GraphPropagation,
                                                   'InverseGraphPropagation': InverseGraphPropagation,
                                                   'GraphLSTM': GraphLSTM,
                                                   'GraphLSTMCell': GraphLSTMCell,
                                                   'custom_mse': custom_mse})

    for img_batch_start in tqdm(range(int(numpy.ceil(len(image_list) / PREDICT_BATCH_SIZE)))):
        batch_img = []
        batch_slic = []
        batch_vertices = []
        batch_neighbors = []
        scale_list = []
        image_names = []
        images_list = []
        for img_index in range(PREDICT_BATCH_SIZE):
            real_index = PREDICT_BATCH_SIZE * img_batch_start + img_index
            image_name = image_list[real_index]
            if image_name is not None:
                # LOAD IMAGES
                image = io.imread(IMAGES_PATH + image_name + ".png")
                images_list.append(image)
                scale_list.append(image.shape)
                image_names.append(image_name)
                img = resize(image, IMAGE_SHAPE, anti_aliasing=True)

                # OBTAIN OTHER USEFUL DATA
                slic = obtain_superpixels(img, N_SUPERPIXELS, SLIC_SIGMA)
                vertices = average_rgb_for_superpixels(img, slic)
                neighbors = get_neighbors(slic, N_SUPERPIXELS)
            else:
                img = numpy.zeros(IMAGE_SHAPE, dtype=float)
                slic = numpy.zeros(SLIC_SHAPE, dtype=float)
                vertices = average_rgb_for_superpixels(img, slic)
                neighbors = get_neighbors(slic, N_SUPERPIXELS)

            # ADD TO BATCH
            batch_img += [img]
            batch_slic += [slic]
            batch_vertices += [vertices]
            batch_neighbors += [neighbors]
        batch_img = numpy.array(batch_img)
        batch_slic = numpy.array(batch_slic)
        batch_vertices = numpy.array(batch_vertices)
        batch_neighbors = numpy.array(batch_neighbors)

        output_vertices = model.predict_on_batch([batch_vertices, batch_neighbors])

        for index, shape in enumerate(scale_list):
            slic_out = batch_slic[index]
            output_image = numpy.zeros(batch_img[index].shape, dtype="uint8")
            for segment_num in range(output_vertices.shape[1]):
                if segment_num not in numpy.unique(batch_slic[index]):
                    break
                mask = numpy.zeros(batch_slic[index, :, :].shape + (3,), dtype="uint8")
                mask[batch_slic[index, :, :] == segment_num] = 255 * output_vertices[index, segment_num, :]
                output_image += mask

            output_image = resize(output_image, shape, anti_aliasing=True)
            output_image = numpy.clip(output_image * 255, 0, 255)
            expected_image = io.imread(VALIDATION_IMAGES + image_names[index] + ".png")
            i = numpy.concatenate((images_list[index], expected_image, output_image), axis=1)
            output = numpy.clip(i, 0, 255)
            output = output.astype(numpy.uint8)
            io.imsave(OUTPUT_PATH + image_names[index] + ".png", output)
