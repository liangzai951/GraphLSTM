from keras.engine import Layer
import tensorflow as tf


class ConfidenceLayer(Layer):
    def __init__(self, n_features=1, **kwargs):
        self.n_features = n_features
        super(ConfidenceLayer, self).__init__(**kwargs)

    def build(self, input_shape): super(ConfidenceLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        shape_out = tuple(inputs.shape[:-1].as_list()) + (self.n_features,)
        return tf.keras.backend.random_uniform(shape=shape_out[1:])

    def compute_output_shape(self, input_shape): return (input_shape[:-1]) + (self.n_features,)
