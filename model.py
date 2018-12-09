import tensorflow as tf
from keras import Input, Model
from keras.layers import LSTM, Conv2D
from keras.utils.vis_utils import plot_model

from layers.ConfidenceLayer import ConfidenceLayer

from layers.Convert2ImageLayer import Convert2ImageLayer
from layers.GraphLSTM import GraphLSTM

IMAGE_SHAPE = (720, 1024, 3)
SLIC_SHAPE = (IMAGE_SHAPE[0], IMAGE_SHAPE[1])
N_SUPERPIXELS = 100
N_FEATURES = 5


def create_model():
    image_input = Input(shape=IMAGE_SHAPE, name="Image")
    slic_input = Input(shape=SLIC_SHAPE, name="SLIC")
    superpixels_input = Input(shape=(N_SUPERPIXELS, 3), name="Vertices")
    neighborhood_input = Input(shape=(N_SUPERPIXELS, N_SUPERPIXELS), name="Neighborhood")

    confidence_map = ConfidenceLayer(N_SUPERPIXELS, name="ConfidenceMap")([image_input, slic_input])
    lstm = GraphLSTM(IMAGE_SHAPE[-1], return_sequences=True, name="LSTM")([superpixels_input, confidence_map, neighborhood_input])
    convert_layer = Convert2ImageLayer(max_segments=N_SUPERPIXELS, name="ToImage")([lstm, slic_input])

    last = Conv2D(IMAGE_SHAPE[-1], kernel_size=1, padding="same", name="OutputConvolution")(convert_layer)

    model = Model(inputs=[image_input,
                          slic_input,
                          superpixels_input,
                          neighborhood_input],
                  outputs=[last, confidence_map, neighborhood_input])
    model.summary()
    # plot_model(model)


if __name__ == '__main__':
    create_model()

    # image --+----> confidence map ok
    #  |      |              |
    #  v      v              v
    # slic -> nodes ----> GraphLSTM -----v
    #  |---------------------+--------> convert2image
