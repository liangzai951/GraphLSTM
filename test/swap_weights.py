import numpy as np
from keras.engine.saving import load_model

from config import RAW_MODEL_PATH, VALIDATION_MODEL
from layers.GraphLSTM import GraphLSTM
from layers.GraphLSTMCell import GraphLSTMCell

def printer(x):
    # print(x)
    x_i = x[0][:, 0:3]
    x_f = x[0][:, 3:6]
    x_c = x[0][:, 6:9]
    x_o = x[0][:, 9:12]
    print("===============================")
    print("Xi: ")
    print(x_i)
    print("Xf: ")
    print(x_f)
    print("Xc: ")
    print(x_c)
    print("Xo: ")
    print(x_o)
    print("===============================")


if __name__ == '__main__':
    raw_model = load_model(VALIDATION_MODEL,
                           custom_objects={'GraphLSTM': GraphLSTM,
                                           'GraphLSTMCell': GraphLSTMCell})
    x = raw_model.get_weights()
    printer(x)

    # TRUE
    # x[0][:, 3:6] = -10.0
    # x[1] = np.zeros(x[1].shape, dtype=np.float32)
    # x[2] = np.zeros(x[2].shape, dtype=np.float32)
    # x[3] = np.zeros(x[3].shape, dtype=np.float32)

    # MODIFIED
    # x[0][:, 0:3] = 0.0
    # x[0][:, 6:9] = 0.0
    # x[0][:, 9:12] = 0.0

    printer(x)
    # raw_model.set_weights(x)
    # raw_model.save(VALIDATION_MODEL)
