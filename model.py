import tensorflow as tf
from keras import Input, Model
from keras.layers import LSTM
from keras.utils.vis_utils import plot_model

from layers.ConfidenceLayer import ConfidenceLayer

from layers.Convert2ImageLayer import Convert2ImageLayer

IMAGE_SHAPE = (720, 1024, 3)
SLIC_SHAPE = (IMAGE_SHAPE[0], IMAGE_SHAPE[1], 1)
N_SUPERPIXELS = 100
N_FEATURES = 5


def create_model():
    image_input = Input(shape=IMAGE_SHAPE, name="Image")
    slic_input = Input(shape=SLIC_SHAPE, name="SLIC")
    superpixels_input = Input(shape=(N_SUPERPIXELS, 3), name="Vertices")

    confidence_map = ConfidenceLayer(N_FEATURES, name="ConfidenceMap")(image_input)
    lstm = LSTM(IMAGE_SHAPE[-1], return_sequences=True, name="LSTM")(superpixels_input)
    concat_layer = Convert2ImageLayer(max_segments=N_SUPERPIXELS, name="ToImage")([lstm, slic_input])

    model = Model(inputs=[image_input, slic_input, superpixels_input], outputs=[confidence_map, concat_layer])
    model.summary()
    # plot_model(model)


if __name__ == '__main__':
    create_model()

    # image --+----> confidence map ok
    #  |      |              |
    #  v      v              v
    # slic -> nodes ----> GraphLSTM -----v
    #  |---------------------+--------> convert2image
