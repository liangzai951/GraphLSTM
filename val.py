import numpy
from keras.engine.saving import load_model
from skimage import io

from config import MODEL_PATH, VALSET_FILE, IMAGES_PATH, N_SUPERPIXELS, \
    SLIC_SIGMA, OUTPUT_PATH
from utils.utils import obtain_superpixels, average_rgb_for_superpixels, \
    get_neighbors

if __name__ == '__main__':
    model = load_model(MODEL_PATH)
    with open(VALSET_FILE) as f:
        image_list = [line for line in f]
    for image_name in image_list:
        img = io.imread(IMAGES_PATH + image_name + ".jpg")
        slic = obtain_superpixels(img, N_SUPERPIXELS, SLIC_SIGMA)
        vertices = average_rgb_for_superpixels(img, slic)
        neighbors = get_neighbors(slic, N_SUPERPIXELS)
        output_vertices = model.predict_on_batch([
            img, slic, vertices, neighbors
        ])
        slic_out = slic
        output_image = numpy.zeros(img.shape, dtype="uint8")
        for segment_num in numpy.unique(slic):
            mask = numpy.zeros(slic.shape, dtype="uint8")
            mask[slic == segment_num] = output_vertices[segment_num]
            output_image += mask
        io.imsave(OUTPUT_PATH+image_name+"jpg")
