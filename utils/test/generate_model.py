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
    lstm_cell1 = GraphLSTMCell(IMAGE_SHAPE[-1],
                               kernel_initializer=init,
                               recurrent_initializer=init,
                               bias_initializer=init)
    lstm_cell2 = GraphLSTMCell(IMAGE_SHAPE[-1],
                               kernel_initializer=init,
                               recurrent_initializer=init,
                               bias_initializer=init)
    return lstm_cell1, lstm_cell2


def generate_model():
    # INPUTS
    image = Input(shape=IMAGE_SHAPE, name="Image", batch_shape=(TRAIN_BATCH_SIZE,) + IMAGE_SHAPE)
    slic = Input(shape=SLIC_SHAPE, name="SLIC", batch_shape=(TRAIN_BATCH_SIZE,) + SLIC_SHAPE)
    superpixels = Input(shape=(N_SUPERPIXELS, IMAGE_SHAPE[2]), name="Vertices", batch_shape=(TRAIN_BATCH_SIZE, N_SUPERPIXELS, IMAGE_SHAPE[2]))
    neighbors = Input(shape=(N_SUPERPIXELS, N_SUPERPIXELS), name="Neighborhood", batch_shape=(TRAIN_BATCH_SIZE, N_SUPERPIXELS, N_SUPERPIXELS))

    # CONFIDENCE MAP
    conv_init = RandomNormal(stddev=0.001)
    conv1 = Conv2D(8, 3, padding='same', kernel_initializer=conv_init,
                   bias_initializer=conv_init)(image)
    conv2 = Conv2D(16, 1, padding='same', kernel_initializer=conv_init,
                   bias_initializer=conv_init)(conv1)
    conv3 = Conv2D(N_FEATURES, 1, padding='same', kernel_initializer=conv_init,
                   bias_initializer=conv_init)(conv2)
    conv4 = Conv2D(1, 1, padding='same', kernel_initializer=conv_init,
                   bias_initializer=conv_init)(conv3)
    confidence = Confidence(N_SUPERPIXELS, name="ConfidenceMap")([conv3, slic])

    # GRAPH PROPAGATION
    graph, reverse, mapping = GraphPropagation(N_SUPERPIXELS, name="GraphPath")([superpixels, confidence, neighbors])

    # MAIN LSTM PART
    lstm_cell1, lstm_cell2 = get_cells()
    lstm1 = GraphLSTM(lstm_cell1, return_sequences=True, name="G-LSTM1", stateful=False)([graph, superpixels, neighbors, mapping, reverse])

    residual1 = add([graph, lstm1])

    lstm2 = GraphLSTM(lstm_cell2, return_sequences=True, name="G-LSTM2", stateful=False)([residual1, superpixels, neighbors, mapping, reverse])

    residual2 = add([residual1, lstm2])

    # INVERSE GRAPH PROPAGATION
    out_vertices = InverseGraphPropagation(name="InvGraphPath")([residual2, reverse])

    out_conv = Conv1D(IMAGE_SHAPE[-1], 1, name="OutputConv0")(out_vertices)
    # out3 = Conv1D(IMAGE_SHAPE[-1], 1, name="OutputConv1")(out_conv)
    # out_2 = Conv1D(IMAGE_SHAPE[-1], 1, name="OutputConv2")(out3)
    # out = Conv1D(IMAGE_SHAPE[-1], 1, name="OutputConv3")(out2)

    # out_m = keras.layers.add([out_2, confidence])

    # out = out_conv
    out = Softmax(axis=-1)(out_conv)

    model = Model(inputs=[image,
                          slic,
                          superpixels,
                          neighbors],
                  outputs=[out, conv4])

    model.summary()

    # PLOT
    plot_model(model, show_shapes=True)

    # OPTIMIZER
    sgd = SGD(lr=0.001, momentum=0.9, decay=0.005, nesterov=False)
    model.compile(sgd, loss=["categorical_crossentropy", custom_mse], metrics=["categorical_accuracy", "acc"])
    model.save("glstm_test.hdf5")
    return model


if __name__ == '__main__':
    generate_model()
