from keras.engine import Layer
import tensorflow as tf
import keras.backend as K


class ConfidenceLayer(Layer):
    def __init__(self, n_segments, **kwargs):
        self.n_segments = n_segments
        self.matrix = None
        super(ConfidenceLayer, self).__init__(**kwargs)

    def build(self, input_shape): super(ConfidenceLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        image_output, slic_output = inputs

        slic_transposed = slic_output - 1
        self.matrix = K.placeholder(shape=(inputs[0].shape[0], self.n_segments, inputs[0].shape[-1]))
        self.matrix *= 0
        self.matrix -= 1
        cycles_list = []
        for cycle in range(self.n_segments):
            # get segment
            segment = image_output*tf.cast(slic_transposed == 0, "float32")
            # get average
            avg = K.sum(segment, axis=[1, 2])/tf.math.count_nonzero(segment, axis=[1, 2], dtype="float32")
            # insert into matrix
            cycles_list.append(avg)
            slic_transposed -= 1
        self.matrix = tf.stack(cycles_list)
        return self.matrix

    def compute_output_shape(self, input_shape): return input_shape[0][0], self.n_segments, input_shape[0][-1]
