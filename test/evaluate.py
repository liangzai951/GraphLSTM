import copy

import numpy
from keras.engine.saving import load_model
import skimage
from skimage.transform import resize
from tqdm import tqdm

from skimage.segmentation import mark_boundaries
from config import *
from layers.GraphLSTM import GraphLSTM
from layers.GraphLSTMCell import GraphLSTMCell
from layers.InverseGraphPropagation import InverseGraphPropagation
from utils.utils import obtain_superpixels, average_rgb_for_superpixels, \
    get_neighbors, sort_values, get_superpixels_index_for_hot_areas

if __name__ == '__main__':
    for t in [2, 3, 5, 7, 9]:
        while len(image_list) % PREDICT_BATCH_SIZE != 0:
            image_list.append(None)
        VALIDATION_IMAGES = IMAGES_PATH = "../data/test/"
        # VALIDATION_MODEL = RAW_MODEL_PATH
        RAW_MODEL_PATH = "glstm_raw{0!s}.hdf5".format(t)
        VALIDATION_MODEL = "glstm{0!s}.hdf5".format(t)
        model = load_model(VALIDATION_MODEL,
                           custom_objects={'GraphLSTM': GraphLSTM,
                                           'GraphLSTMCell': GraphLSTMCell,
                                           'InverseGraphPropagation': InverseGraphPropagation})
        raw_model = load_model(RAW_MODEL_PATH,
                               custom_objects={'GraphLSTM': GraphLSTM,
                                               'GraphLSTMCell': GraphLSTMCell,
                                               'InverseGraphPropagation': InverseGraphPropagation})

        for img_batch_start in tqdm(
                range(int(numpy.ceil(len(image_list) / PREDICT_BATCH_SIZE)))):
            batch_img = []
            batch_vertices = []
            batch_slic = []
            batch_neighbors = []
            batch_expected = []

            batch_inputs = [[] for _ in range(t)]
            batch_indexes = [[] for _ in range(t)]
            batch_r_indexes = [[] for _ in range(t)]

            scale_list = []
            image_names = []
            images_list = []
            for img_index in range(PREDICT_BATCH_SIZE):
                real_index = PREDICT_BATCH_SIZE * img_batch_start + img_index
                image_name = image_list[real_index]
                if image_name is not None:
                    # LOAD IMAGES
                    image = skimage.io.imread(IMAGES_PATH + image_name + ".png")
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
                areas = get_superpixels_index_for_hot_areas(slic)
                expected = copy.deepcopy(vertices)
                for i in expected:
                    i.append(0.0)

                green_near_red_indices = []
                for i, v in enumerate(vertices):
                    if v[0] != 1.0:
                        continue
                    neighborhood_indexes = numpy.where(neighbors[i] == 1)[0]
                    if any(vertices[n][1] == 1.0 for n in
                           neighborhood_indexes):
                        green_near_red_indices.append(i)

                for i in green_near_red_indices:
                    expected[i] = [0.0] * len(expected[i])
                    expected[i][-1] = 1.0

                for paths in range(t):
                    vertex_index = areas[paths]
                    path, mapping, r_mapping = sort_values(vertices, neighbors,
                                                           vertex_index,
                                                           mode="bfs")
                    batch_inputs[paths].append(path)
                    batch_indexes[paths].append(mapping)
                    batch_r_indexes[paths].append(r_mapping)

                # ADD TO BATCH
                batch_vertices += [vertices]
                batch_neighbors += [neighbors]
                batch_slic += [slic]
                batch_img += [img]
                batch_expected += [expected]
                batch_vertices = numpy.array(batch_vertices)
                batch_neighbors = numpy.array(batch_neighbors)
            batch_inputs = [numpy.array(i) for i in batch_inputs]
            batch_indexes = [numpy.array(i) for i in batch_indexes]
            batch_r_indexes = [numpy.array(i) for i in batch_r_indexes]

            batch_vertices = numpy.array(batch_vertices)
            batch_neighbors = numpy.array(batch_neighbors)
            batch_slic = numpy.array(batch_slic)
            batch_expected = numpy.array(batch_expected)

            output_vertices = model.predict_on_batch(
                [batch_vertices, batch_neighbors] +
                batch_inputs +
                batch_indexes +
                batch_r_indexes)
            raw_output_vertices = raw_model.predict_on_batch(
                [batch_vertices, batch_neighbors] +
                batch_inputs +
                batch_indexes +
                batch_r_indexes)
            # raw_output_vertices[:, :, :] = raw_output_vertices[:,
            #                                batch_r_indexes[0]]
            # output_vertices[:, :, :] = output_vertices[:, batch_r_indexes[0]]
            for index, shape in enumerate(scale_list):
                slic_out = batch_slic[index]
                output_image = numpy.zeros(batch_img[index].shape, dtype="uint8")
                expected_image = numpy.zeros(batch_img[index].shape, dtype="uint8")
                raw_output_image = numpy.zeros(batch_img[index].shape, dtype="uint8")
                for segment_num in range(output_vertices.shape[1]):
                    if segment_num not in numpy.unique(batch_slic[index]):
                        break
                    mask = numpy.zeros(batch_slic[index, :, :].shape + (3,),
                                       dtype="uint8")
                    color = list(output_vertices[index, segment_num, :])
                    chosen_color = color.index(max(color))
                    color = numpy.zeros(3, dtype=numpy.float32)
                    if chosen_color == 3:
                        color[0] = 1.0
                        color[1] = 1.0
                    else:
                        color[chosen_color] = 1.0
                    mask[batch_slic[index, :,
                         :] == segment_num] = 255 * color
                    output_image += mask
                for segment_num in range(raw_output_vertices.shape[1]):
                    if segment_num not in numpy.unique(batch_slic[index]):
                        break
                    mask = numpy.zeros(batch_slic[index, :, :].shape + (3,),
                                       dtype="uint8")
                    color = list(raw_output_vertices[index, segment_num, :])
                    chosen_color = color.index(max(color))
                    color = numpy.zeros(3, dtype=numpy.float32)
                    if chosen_color == 3:
                        color[0] = 1.0
                        color[1] = 1.0
                    else:
                        color[chosen_color] = 1.0
                    mask[batch_slic[index, :,
                         :] == segment_num] = 255 * color
                    raw_output_image += mask
                for segment_num in range(batch_expected.shape[1]):
                    if segment_num not in numpy.unique(batch_slic[index]):
                        break
                    mask = numpy.zeros(batch_slic[index, :, :].shape + (3,),
                                       dtype="uint8")
                    color = list(batch_expected[index, segment_num, :])
                    chosen_color = color.index(max(color))
                    color = numpy.zeros(3, dtype=numpy.float32)
                    if chosen_color == 3:
                        color[0] = 1.0
                        color[1] = 1.0
                    else:
                        color[chosen_color] = 1.0
                    mask[batch_slic[index, :,
                         :] == segment_num] = 255 * color
                    expected_image += mask

                output_image = resize(output_image, shape, anti_aliasing=True)
                output_image = mark_boundaries(output_image, slic_out)
                output_image = numpy.clip(output_image * 255, 0, 255)
                expected_image = resize(expected_image, shape, anti_aliasing=True)
                expected_image = mark_boundaries(expected_image, slic_out)
                expected_image = numpy.clip(expected_image * 255, 0, 255)
                raw_output_image = resize(raw_output_image, shape,
                                          anti_aliasing=True)
                raw_output_image = mark_boundaries(raw_output_image, slic_out)
                raw_output_image = numpy.clip(raw_output_image * 255, 0, 255)
                # expected_image = io.imread(
                #     VALIDATION_IMAGES + image_names[index] + ".png")
                img_in = mark_boundaries(skimage.img_as_float(images_list[index]),
                                         slic_out)
                img_in = numpy.clip(img_in * 255, 0, 255)
                i = numpy.concatenate(
                    (img_in, expected_image, raw_output_image, output_image),
                    axis=1)
                output = numpy.clip(i, 0, 255)
                output = output.astype(numpy.uint8)
                skimage.io.imsave(OUTPUT_PATH + image_names[index] + "_{0!s}.png".format(t), output)
