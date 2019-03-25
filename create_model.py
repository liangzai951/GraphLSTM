from keras import Input, Model
from keras.layers import Conv2D, Conv1D, LSTMCell
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model

from config import *
from layers.ConfidenceLayer import Confidence
from layers.GraphLSTM import GraphLSTM
from layers.GraphLSTMCell import GraphLSTMCell
from layers.GraphPropagation import GraphPropagation
from layers.InverseGraphPropagation import InverseGraphPropagation


def create_model():
    # INPUTS
    image = Input(shape=IMAGE_SHAPE, name="Image", batch_shape=(TRAIN_BATCH_SIZE,) + IMAGE_SHAPE)
    slic = Input(shape=SLIC_SHAPE, name="SLIC", batch_shape=(TRAIN_BATCH_SIZE,) + SLIC_SHAPE)
    superpixels = Input(shape=(N_SUPERPIXELS, IMAGE_SHAPE[2]), name="Vertices", batch_shape=(TRAIN_BATCH_SIZE, N_SUPERPIXELS, IMAGE_SHAPE[2]))
    neighbors = Input(shape=(N_SUPERPIXELS, N_SUPERPIXELS), name="Neighborhood", batch_shape=(TRAIN_BATCH_SIZE, N_SUPERPIXELS, N_SUPERPIXELS))

    # IMAGE CONVOLUTION
    conv1 = Conv2D(8, 5, padding='same')(image)
    conv2 = Conv2D(16, 3, padding='same')(conv1)
    conv3 = Conv2D(32, 3, padding='same')(conv2)
    conv4 = Conv2D(1, 3, padding='same')(conv3)

    # CONFIDENCE MAP
    confidence = Confidence(N_SUPERPIXELS, name="ConfidenceMap", trainable=False)([conv3, slic])

    # GRAPH PROPAGATION
    graph, reverse = GraphPropagation(N_SUPERPIXELS, name="GraphPath", trainable=False)([superpixels, confidence, neighbors])

    # MAIN LSTM PART
    lstm_cell = GraphLSTMCell(IMAGE_SHAPE[-1])
    lstm = GraphLSTM(lstm_cell, return_sequences=True, name="G-LSTM", stateful=True)(graph)
    # lstm = GraphLSTM(IMAGE_SHAPE[-1], return_sequences=True, name="G-LSTM", stateful=True)([graph, superpixels, neighbors, mapping])
    # lstm2 = LSTM(IMAGE_SHAPE[-1], return_sequences=True, name="G-LSTM2")(lstm)

    # INVERSE GRAPH PROPAGATION
    out_vertices = InverseGraphPropagation(name="InvGraphPath", trainable=False)([lstm, reverse])

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
    sgd = SGD(momentum=0.9, decay=0.0005)
    model.compile(sgd, loss="mse", metrics=["acc"])
    model.save(MODEL_PATH)
    return model


if __name__ == '__main__':
    create_model()
