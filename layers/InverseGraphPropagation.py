from keras.engine import Layer
import keras.backend as K
import tensorflow as tf


class InverseGraphPropagation(Layer):
    def __init__(self, **kwargs):
        super(InverseGraphPropagation, self).__init__(**kwargs)

    def build(self, input_shape):
        super(InverseGraphPropagation, self).build(input_shape)

    def call(self, inputs, **kwargs):
        tf.logging.log(tf.logging.ERROR, inputs)
        def inv_mapper(input_tuple):
            vertices = input_tuple[0]
            reverse_map = input_tuple[1]

            tf.logging.log(tf.logging.ERROR, vertices)
            tf.logging.log(tf.logging.ERROR, reverse_map)
            new_vertices = K.placeholder(vertices.shape)
            # for i, index in enumerate(reverse_map):
            #     new_vertices[i] = vertices[index]
            return K.stack(new_vertices)

        return K.map_fn(inv_mapper, (inputs[0], inputs[1]), dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        self.__verify_input_shape(input_shape)
        vertices, reverse_map = input_shape
        return vertices[0], vertices[1], vertices[2]

    @staticmethod
    def __verify_input_shape(input_shape):
        assert isinstance(input_shape, list)
        vertices, reverse_map = input_shape
        assert vertices[1] == reverse_map[1]
