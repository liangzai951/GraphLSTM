import keras.backend as K
import keras
from keras import Input, Model
from keras.initializers import RandomUniform, RandomNormal
from keras.layers import Conv1D, add, Lambda, Conv2D, Softmax
from keras.optimizers import SGD
from keras.utils import plot_model

from config import IMAGE_SHAPE, TRAIN_BATCH_SIZE, SLIC_SHAPE, N_SUPERPIXELS, \
    custom_mse, N_FEATURES
from layers.ConfidenceLayer import Confidence
from layers.GraphLSTM import GraphLSTM
from layers.GraphLSTMCell import GraphLSTMCell
from layers.GraphPropagation import GraphPropagation
from layers.InverseGraphPropagation import InverseGraphPropagation


def get_cells():
    init = RandomUniform(minval=-0.1, maxval=0.1, seed=None)
    lstm_cell1 = GraphLSTMCell(N_FEATURES,
                               kernel_initializer=init,
                               recurrent_initializer=init,
                               bias_initializer=init)
    lstm_cell2 = GraphLSTMCell(N_FEATURES,
                               kernel_initializer=init,
                               recurrent_initializer=init,
                               bias_initializer=init)
    return lstm_cell1, lstm_cell2


def generate_model():
    # INPUTS
    superpixels = Input(shape=(N_SUPERPIXELS, IMAGE_SHAPE[2]), name="Vertices",
                        batch_shape=(
                        TRAIN_BATCH_SIZE, N_SUPERPIXELS, IMAGE_SHAPE[2]))
    neighbors = Input(shape=(N_SUPERPIXELS, N_SUPERPIXELS),
                      name="Neighborhood", batch_shape=(
        TRAIN_BATCH_SIZE, N_SUPERPIXELS, N_SUPERPIXELS))

    # CONFIDENCE MAP
    conv_init = RandomNormal(stddev=0.001)
    conv1 = Conv1D(8, 3, padding='same', kernel_initializer=conv_init,
                   bias_initializer=conv_init)(superpixels)
    conv2 = Conv1D(32, 3, padding='same', kernel_initializer=conv_init,
                   bias_initializer=conv_init)(conv1)
    conv2a = Conv1D(96, 3, padding='same', kernel_initializer=conv_init,
                    bias_initializer=conv_init)(conv2)
    conv3 = Conv1D(N_FEATURES, 3, padding='same', kernel_initializer=conv_init,
                   bias_initializer=conv_init)(conv2a)
    conv_confidence = Conv1D(N_FEATURES, 1, padding='same',
                             kernel_initializer=conv_init,
                             bias_initializer=conv_init)(conv3)

    # GRAPH PROPAGATION
    graph, reverse, mapping = GraphPropagation(N_SUPERPIXELS,
                                               name="GraphPath")(
        [conv3, conv_confidence, neighbors])

    # MAIN LSTM PART
    lstm_cell1, lstm_cell2 = get_cells()
    lstm1 = GraphLSTM(lstm_cell1, return_sequences=True, name="G-LSTM1",
                      stateful=False)(
        [graph, conv3, neighbors, mapping, reverse])

    residual1 = add([graph, lstm1])

    lstm2 = GraphLSTM(lstm_cell2, return_sequences=True, name="G-LSTM2",
                      stateful=False)(
        [residual1, conv3, neighbors, mapping, reverse])

    residual2 = add([residual1, lstm2])

    # INVERSE GRAPH PROPAGATION
    out_vertices = InverseGraphPropagation(name="InvGraphPath")(
        [residual2, reverse])

    out3 = Conv1D(N_FEATURES, 1, name="OutputConv1")(out_vertices)

    out_m = keras.layers.multiply([out3, conv_confidence])

    out_conv = Conv1D(IMAGE_SHAPE[-1], 1, name="OutputConv0")(out_m)
    # out3 = Conv1D(IMAGE_SHAPE[-1], 1, name="OutputConv1")(out_conv)
    # out_2 = Conv1D(IMAGE_SHAPE[-1], 1, name="OutputConv2")(out3)
    # out = Conv1D(IMAGE_SHAPE[-1], 1, name="OutputConv3")(out2)

    # out_m = keras.layers.add([out_vertices, conv_confidence])

    out = out_conv
    # out = Softmax(axis=-1)(out_conv)

    model = Model(inputs=[superpixels,
                          neighbors],
                  outputs=[out])

    model.summary()

    # PLOT
    plot_model(model, show_shapes=True)

    # OPTIMIZER
    sgd = SGD(lr=0.001, momentum=0.9, decay=0.005, nesterov=False)
    model.compile(sgd, loss=custom_mse,
                  metrics=["acc"])
    model.save("glstm_test.hdf5")
    return model


if __name__ == '__main__':
    generate_model()
