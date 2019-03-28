from keras import Input, Model
from keras.initializers import RandomUniform, RandomNormal
from keras.layers import Conv2D, Conv1D
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model

from config import *
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

def create_model():
    # INPUTS
    image = Input(shape=IMAGE_SHAPE, name="Image", batch_shape=(TRAIN_BATCH_SIZE,) + IMAGE_SHAPE)
    slic = Input(shape=SLIC_SHAPE, name="SLIC", batch_shape=(TRAIN_BATCH_SIZE,) + SLIC_SHAPE)
    superpixels = Input(shape=(N_SUPERPIXELS, IMAGE_SHAPE[2]), name="Vertices", batch_shape=(TRAIN_BATCH_SIZE, N_SUPERPIXELS, IMAGE_SHAPE[2]))
    neighbors = Input(shape=(N_SUPERPIXELS, N_SUPERPIXELS), name="Neighborhood", batch_shape=(TRAIN_BATCH_SIZE, N_SUPERPIXELS, N_SUPERPIXELS))

    # IMAGE CONVOLUTION
    conv_init = RandomNormal(stddev=0.001)
    conv1 = Conv2D(8, 3, padding='same', kernel_initializer=conv_init, bias_initializer=conv_init)(image)
    conv2 = Conv2D(16, 3, padding='same', kernel_initializer=conv_init, bias_initializer=conv_init)(conv1)
    conv3 = Conv2D(N_FEATURES, 1, padding='same', kernel_initializer=conv_init, bias_initializer=conv_init)(conv2)
    conv4 = Conv2D(1, 1, padding='same', kernel_initializer=conv_init, bias_initializer=conv_init)(conv3)

    # CONFIDENCE MAP
    confidence = Confidence(N_SUPERPIXELS, name="ConfidenceMap", trainable=False)([conv3, slic])

    # GRAPH PROPAGATION
    graph, reverse, mapping = GraphPropagation(N_SUPERPIXELS, name="GraphPath", trainable=False)([superpixels, confidence, neighbors])

    # MAIN LSTM PART
    lstm_cell1, lstm_cell2 = get_cells()
    lstm1 = GraphLSTM(lstm_cell1, return_sequences=True, name="G-LSTM1", stateful=True)([graph, superpixels, neighbors, mapping, reverse])
    lstm2 = GraphLSTM(lstm_cell2, return_sequences=True, name="G-LSTM2", stateful=True)([lstm1, superpixels, neighbors, mapping, reverse])

    # INVERSE GRAPH PROPAGATION
    out_vertices = InverseGraphPropagation(name="InvGraphPath", trainable=False)([lstm2, reverse])

    out = Conv1D(IMAGE_SHAPE[-1], 1, name="OutputConv")(out_vertices)
    # out = out_vertices

    # # TO IMAGE CONVERSION
    # to_image = Convert2Image(max_segments=N_SUPERPIXELS, name="ToImage")([out_vertices, slic])
    # # OUTPUT
    # output = Conv2D(IMAGE_SHAPE[-1], kernel_size=1, padding="same", name="OutputConvolution")(to_image)
    # model = Model(inputs=[image,
    #                       slic,
    #                       superpixels,
    #                       neighbors],
    #               outputs=[output])

    model = Model(inputs=[image,
                          slic,
                          superpixels,
                          neighbors],
                  outputs=[out, conv4])

    model.summary()

    # PLOT
    plot_model(model, show_shapes=True)

    # OPTIMIZER
    sgd = SGD(lr=0.001, momentum=0.9, decay=0.0005, nesterov=True)
    model.compile(sgd, loss="mse", metrics=["acc"])
    model.save(MODEL_PATH)
    return model


if __name__ == '__main__':
    create_model()
