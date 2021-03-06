import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.engine import Layer


class GraphPropagation(Layer):
    def __init__(self, n_vertices, **kwargs):
        self.n_vertices = n_vertices
        super(GraphPropagation, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GraphPropagation, self).build(input_shape)

    def call(self, inputs, **kwargs):

        def graph_propagation(input_tuple):
            confidence, neighborhood, vertices_batch = input_tuple
            confidence = K.max(confidence, axis=-1, keepdims=False)
            neighborhood = K.mean(neighborhood, axis=-1)
            confidence = K.tf.add(confidence, neighborhood)
            maping = K.tf.argsort(confidence, axis=0, direction='DESCENDING')
            new_vertices = K.tf.gather(vertices_batch, maping)
            reverse_mapping = K.tf.argsort(maping, axis=0, direction='ASCENDING')
            return new_vertices, reverse_mapping, maping

        vertices = inputs[0]
        confidence_map = inputs[1]
        neighborhood_matrix = inputs[2]
        v, reverse_mapping, mapping = K.map_fn(graph_propagation, (confidence_map, neighborhood_matrix, vertices), dtype=(tf.float32, tf.int32, tf.int32))
        return [v, reverse_mapping, mapping]

    def compute_output_shape(self, input_shape):
        self.__verify_input_shape(input_shape)
        vertices, confidence_map, neighbors = input_shape
        return [vertices, (vertices[0], vertices[1]), (vertices[0], vertices[1])]

    @staticmethod
    def __verify_input_shape(input_shape):
        assert isinstance(input_shape, list)
        vertices, confidence_map, neighbors = input_shape
        assert vertices[1] == confidence_map[1] == neighbors[1] == neighbors[2]

    def get_config(self):
        config = {'n_vertices': self.n_vertices}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
