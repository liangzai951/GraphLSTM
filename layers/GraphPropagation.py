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
            vertices = input_tuple[0]
            confidence_map = K.max(input_tuple[1], axis=-1, keepdims=False)
            neighbors = input_tuple[2]

            unvisited = {i for i in range(vertices.shape[0])}
            new_vertices = []
            reverse_map = []
            neighborhood = set()

            first_confident = np.argmax(confidence_map, axis=0)
            new_vertices.append(vertices[first_confident])
            reverse_map.append(tf.convert_to_tensor(first_confident))
            for i in np.where(neighbors[first_confident] == 1.0)[0]:
                if i in unvisited:
                    neighborhood.add(i)
            confidence_map = confidence_map * tf.cast(confidence_map != confidence_map[first_confident], "float32")
            unvisited.discard(first_confident)
            num = 1
            while unvisited and num < self.n_vertices:
                if neighborhood:
                    first_confident = sorted([(confidence_map[i], i) for i in neighborhood], key=lambda x: x[0])[-1][-1]
                    neighborhood.remove(first_confident)
                else:
                    first_confident = np.argmax(confidence_map, axis=0)
                if first_confident in unvisited:
                    new_vertices.append(vertices[first_confident])
                    reverse_map.append(tf.convert_to_tensor(first_confident))
                    for i in np.where(neighbors[first_confident] == 1.0)[0]:
                        if i in unvisited:
                            neighborhood.add(i)
                    confidence_map = confidence_map * tf.cast(confidence_map != confidence_map[first_confident], "float32")
                    unvisited.discard(first_confident)
                num += 1

            return K.stack(new_vertices), K.stack(reverse_map)

        v, m = K.map_fn(graph_propagation, (inputs[0], inputs[1], inputs[2]), dtype=(tf.float32, tf.int32))
        return [v, m]

    def compute_output_shape(self, input_shape):
        self.__verify_input_shape(input_shape)
        vertices, confidence_map, neighbors = input_shape
        return [vertices, (vertices[0], vertices[1])]

    @staticmethod
    def __verify_input_shape(input_shape):
        assert isinstance(input_shape, list)
        vertices, confidence_map, neighbors = input_shape
        assert vertices[1] == confidence_map[1] == neighbors[1] == neighbors[2]

    def get_config(self):
        config = {'n_vertices': self.n_vertices}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
