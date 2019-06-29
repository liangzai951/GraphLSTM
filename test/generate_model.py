from keras import Input, Model
from keras.initializers import RandomUniform
from keras.layers import Softmax, Dense, Concatenate, Dropout
from keras.optimizers import SGD, RMSprop
from keras.utils import plot_model

from config import IMAGE_SHAPE, TRAIN_BATCH_SIZE, N_SUPERPIXELS, N_FEATURES, \
    INPUT_PATHS, RAW_MODEL_PATH
from layers.GraphLSTM import GraphLSTM
from layers.GraphLSTMCell import GraphLSTMCell
from layers.InverseGraphPropagation import InverseGraphPropagation


def generate_model():
    for k in [2, 3, 5, 7, 9]:
        INPUT_PATHS = k
        init = RandomUniform(minval=-0.1, maxval=0.1, seed=None)

        vertices = [Input(shape=(N_SUPERPIXELS, IMAGE_SHAPE[2]),
                          name="Vertices",
                          batch_shape=(TRAIN_BATCH_SIZE,
                                       N_SUPERPIXELS,
                                       IMAGE_SHAPE[2]))]
        neighbors = [Input(shape=(N_SUPERPIXELS, N_SUPERPIXELS),
                           name="Neighborhood",
                           batch_shape=(TRAIN_BATCH_SIZE,
                                        N_SUPERPIXELS,
                                        N_SUPERPIXELS))]
        inputs = [Input(shape=(N_SUPERPIXELS, IMAGE_SHAPE[2]),
                        name="Vertices_{0!s}".format(i),
                        batch_shape=(TRAIN_BATCH_SIZE,
                                     N_SUPERPIXELS,
                                     IMAGE_SHAPE[2]))
                  for i in range(INPUT_PATHS)]
        indexes = [Input(shape=(N_SUPERPIXELS,),
                         batch_shape=(TRAIN_BATCH_SIZE, N_SUPERPIXELS),
                         name="Index_{0!s}".format(i), dtype="int32")
                   for i in range(INPUT_PATHS)]
        r_indexes = [Input(shape=(N_SUPERPIXELS,),
                           batch_shape=(TRAIN_BATCH_SIZE, N_SUPERPIXELS),
                           name="ReverseIndex_{0!s}".format(i), dtype="int32")
                     for i in range(INPUT_PATHS)]

        cells = [GraphLSTMCell(N_FEATURES, kernel_initializer=init,
                               recurrent_initializer=init,
                               bias_initializer=init) for _ in range(INPUT_PATHS)]
        lstms = [GraphLSTM(cells[i], return_sequences=True,
                           name="G-LSTM_{0!s}".format(i), stateful=False)
                 ([inputs[i], vertices[0], neighbors[0], indexes[i], r_indexes[i]])
                 for i in range(INPUT_PATHS)]

        inverse = [InverseGraphPropagation()([lstms[i], r_indexes[i]]) for i in range(INPUT_PATHS)]

        concat = Concatenate(axis=-1)(inverse)
        # drop0 = Dropout(0.5)(lstms[0])
        # d1 = Dense(int(INPUT_PATHS * N_FEATURES))(drop0)
        # drop1 = Dropout(0.4)(concat)
        # d2 = Dense(int(INPUT_PATHS * N_FEATURES))(drop1)
        d3 = Dense(N_FEATURES)(concat)

        soft = Softmax()(d3)

        model = Model(inputs=vertices + neighbors + inputs + indexes + r_indexes,
                      outputs=[soft])

        model.summary()

        # PLOT
        # plot_model(model, show_shapes=True)

        # OPTIMIZER
        # sgd = SGD(lr=0.001, momentum=0.9, decay=0.005, nesterov=False)
        rms = RMSprop()
        model.compile(rms, loss="categorical_crossentropy", metrics=["acc"])
        model.save("glstm_raw{0!s}.hdf5".format(k))


# return model


if __name__ == '__main__':
    generate_model()
