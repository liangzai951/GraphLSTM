from keras.engine import Layer
import tensorflow as tf
import keras.backend as K


class Confidence(Layer):
    def __init__(self, n_segments, **kwargs):
        self.n_segments = n_segments
        super(Confidence, self).__init__(**kwargs)

    def build(self, input_shape): super(Confidence, self).build(input_shape)

    def call(self, inputs, **kwargs):
        def mapper(inp, n_segments):
            image = inp[0]
            slic_output = inp[1]
            slic_transposed = slic_output - 1
            cycles_list = []
            for cycle in range(n_segments):
                # get segment
                mask = tf.cast(slic_transposed == 0, "float32")
                segment = image * mask
                # get average
                avg = K.sum(segment, axis=[0, 1])/tf.math.count_nonzero(segment, axis=[0, 1], dtype="float32")
                # insert into matrix
                cycles_list.append(avg)
                slic_transposed -= 1
            return K.stack(cycles_list)
        r = K.map_fn(lambda x: mapper(x, self.n_segments), (inputs[0], inputs[1]), dtype=tf.float32)
        # tf.logging.log(tf.logging.ERROR, r)
        return r

    def compute_output_shape(self, input_shape): return input_shape[0][0], self.n_segments, input_shape[0][-1]

    def get_config(self):
        config = {'n_segments': self.n_segments}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
