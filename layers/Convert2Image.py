import tensorflow as tf
import keras.backend as K
from keras.engine import Layer


class Convert2Image(Layer):
    def __init__(self, max_segments, **kwargs):
        super(Convert2Image, self).__init__(**kwargs)
        self.matrix = None
        self.max_segments = max_segments

    def build(self, input_shape):
        super(Convert2Image, self).build(input_shape)

    def call(self, inputs, **kwargs):
        graph_lstm_output, slic_output = inputs

        slic_transposed = slic_output - 1
        self.matrix = K.placeholder(shape=tuple(slic_output.shape.as_list()) + (graph_lstm_output.shape[-1],))
        self.matrix *= 0
        for cycle in range(self.max_segments):
            self.matrix = self.matrix[:, :, :, :] + graph_lstm_output[:, cycle, :] * tf.cast(slic_transposed == 0, "float32")
            slic_transposed -= 1
        return self.matrix

    def compute_output_shape(self, input_shape):
        self.__verify_input_shape(input_shape)
        graph_lstm_output, slic_output = input_shape
        return slic_output + (graph_lstm_output[-1],)

    @staticmethod
    def __verify_input_shape(input_shape): assert isinstance(input_shape, list)
