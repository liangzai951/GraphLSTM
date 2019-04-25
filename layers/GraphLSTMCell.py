from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from keras import backend as K
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import Layer
from keras.layers.recurrent import _generate_dropout_mask
import tensorflow as tf


class GraphLSTMCell(Layer):
    def __init__(self, units,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        super(GraphLSTMCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.recurrent_activation = activations.get(recurrent_activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.unit_forget_bias = unit_forget_bias

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.implementation = implementation
        self.state_size = (self.units, self.units)
        self.output_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        self.W = self.add_weight(shape=(input_dim, self.units * 4),
                                 name='kernel',
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint)
        self.U = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        self.Un = self.add_weight(
            shape=(self.units, self.units * 4),
            name='recurrent_kernel_neighbors',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)

        if self.use_bias:
            if self.unit_forget_bias:
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([
                        self.bias_initializer((self.units,), *args, **kwargs),
                        initializers.Ones()((self.units,), *args, **kwargs),
                        self.bias_initializer((self.units * 2,), *args,
                                              **kwargs),
                    ])
            else:
                bias_initializer = self.bias_initializer
            self.bias = self.add_weight(shape=(self.units * 4,),
                                        name='bias',
                                        initializer=bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.W_i = self.W[:, :self.units]
        self.W_f = self.W[:, self.units: self.units * 2]
        self.W_c = self.W[:, self.units * 2: self.units * 3]
        self.W_o = self.W[:, self.units * 3:]

        self.U_i = self.U[:, :self.units]
        self.U_f = (self.U[:, self.units: self.units * 2])
        self.U_c = (self.U[:, self.units * 2: self.units * 3])
        self.U_o = self.U[:, self.units * 3:]

        self.Un_i = self.Un[:, :self.units]
        self.Un_f = (self.Un[:, self.units: self.units * 2])
        self.Un_c = (self.Un[:, self.units * 2: self.units * 3])
        self.Un_o = self.Un[:, self.units * 3:]

        if self.use_bias:
            self.bias_i = self.bias[:self.units]
            self.bias_f = self.bias[self.units: self.units * 2]
            self.bias_c = self.bias[self.units * 2: self.units * 3]
            self.bias_o = self.bias[self.units * 3:]
        else:
            self.bias_i = None
            self.bias_f = None
            self.bias_c = None
            self.bias_o = None
        self.built = True

    def call(self, inputs, states, time, constants=None, training=None, **kwargs):
        old_vertices, neighbors, mapping, reverse_mapping = constants
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training,
                count=4)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(states[0]),
                self.recurrent_dropout,
                training=training,
                count=4)

        # dropout matrices for input units
        dp_mask = self._dropout_mask
        # dropout matrices for recurrent units
        rec_dp_mask = self._recurrent_dropout_mask

        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        h_tm2 = kwargs['previous_state'][0]
        c_tm2 = kwargs['previous_state'][1]

        previous_position = reverse_mapping[:, time]
        c = tf.convert_to_tensor(numpy.arange(K.int_shape(previous_position)[0]), dtype=tf.int32)
        previous_position_4_gather = K.tf.stack([c, previous_position], axis=-1)

        ng_rows = K.tf.gather_nd(neighbors, previous_position_4_gather)

        def sum_rows(input_ng):
            ng_row, batch_map = input_ng
            ng_num = K.sum(ng_row, axis=-1)
            batch_ng = K.tf.where(K.equal(ng_row, 1))[:, 0]
            current_positions = K.gather(batch_map, batch_ng)

            def sum_unknown(input_time):
                return tf.cond(input_time < time, lambda: h_tm1, lambda: h_tm2)

            tmp_states = K.map_fn(sum_unknown, current_positions, dtype=tf.float32)
            tmp_states = K.tf.div_no_nan(K.sum(tmp_states, axis=[0, 1]), ng_num)
            return tmp_states

        ngs = K.map_fn(sum_rows, (ng_rows, mapping), dtype=tf.float32)

        if self.implementation == 1:
            if 0 < self.dropout < 1.:
                inputs_i = inputs * dp_mask[0]
                inputs_f = inputs * dp_mask[1]
                inputs_c = inputs * dp_mask[2]
                inputs_o = inputs * dp_mask[3]
            else:
                inputs_i = inputs
                inputs_f = inputs
                inputs_c = inputs
                inputs_o = inputs

            x_i = K.dot(inputs_i, self.W_i)
            x_f = K.dot(inputs_f, self.W_f)
            x_f_avg = K.dot(inputs_f, self.W_f)
            x_c = K.dot(inputs_c, self.W_c)
            x_o = K.dot(inputs_o, self.W_o)

            if self.use_bias:
                x_i = K.bias_add(x_i, self.bias_i)
                x_f = K.bias_add(x_f, self.bias_f)
                x_f_avg = K.bias_add(x_f_avg, self.bias_f)
                x_c = K.bias_add(x_c, self.bias_c)
                x_o = K.bias_add(x_o, self.bias_o)

            if 0 < self.recurrent_dropout < 1.:
                h_tm1_i = h_tm1 * rec_dp_mask[0]
                h_tm1_f = h_tm1 * rec_dp_mask[1]
                h_tm1_c = h_tm1 * rec_dp_mask[2]
                h_tm1_o = h_tm1 * rec_dp_mask[3]
            else:
                h_tm1_i = h_tm1
                h_tm1_f = h_tm1
                h_tm1_c = h_tm1
                h_tm1_o = h_tm1
            i = x_i + K.dot(h_tm1_i, self.U_i) + K.dot(ngs, self.Un_i)
            f_avg = x_f_avg + K.dot(h_tm1, self.Un_f)
            f = x_f + K.dot(h_tm1_f, self.U_f)
            c = x_c + K.dot(h_tm1_c, self.U_c) + K.dot(ngs, self.Un_c)
            o = x_o + K.dot(h_tm1_o, self.U_o) + K.dot(ngs, self.Un_o)

            i = self.recurrent_activation(i)
            f_avg = self.recurrent_activation(f_avg)
            f = self.recurrent_activation(f)
            o = self.recurrent_activation(o)
            c = self.activation(c)
        else:
            if 0. < self.dropout < 1.:
                inputs *= dp_mask[0]
            z = K.dot(inputs, self.W)
            if 0. < self.recurrent_dropout < 1.:
                h_tm1 *= rec_dp_mask[0]
            if self.use_bias:
                z = K.bias_add(z, self.bias)

            i = z[:, :self.units]
            f_avg = z[:, self.units: 2 * self.units]
            f = z[:, self.units: 2 * self.units]
            c = z[:, 2 * self.units: 3 * self.units]
            o = z[:, 3 * self.units:]

            i += K.dot(h_tm1, self.U_i) + K.dot(ngs, self.Un_i)
            f_avg += K.dot(h_tm1, self.Un_f)
            f += K.dot(h_tm1, self.U_f)
            o += K.dot(h_tm1, self.U_o) + K.dot(ngs, self.Un_o)
            c += K.dot(h_tm1, self.U_c) + K.dot(ngs, self.Un_c)

            i = self.recurrent_activation(i)
            f_avg = self.recurrent_activation(f_avg)
            f = self.recurrent_activation(f)
            o = self.recurrent_activation(o)
            c = self.activation(c)

        def sum_memories(input_ng):
            ng_row, batch_map = input_ng
            ng_num = K.sum(ng_row, axis=-1)
            batch_ng = K.tf.where(K.equal(ng_row, 1))[:, 0]
            current_positions = K.gather(batch_map, batch_ng)

            def sum_unknown_memories(input_time):
                return tf.cond(input_time < time, lambda: f_avg * c_tm1, lambda: f_avg * c_tm2)

            tmp_states = K.map_fn(sum_unknown_memories, current_positions, dtype=tf.float32)
            tmp_states = K.tf.div_no_nan(K.sum(tmp_states, axis=[0, 1]), ng_num)
            return tmp_states

        memory = K.map_fn(sum_memories, (ng_rows, mapping), dtype=tf.float32)
        memory += f * c_tm1 + i * c

        h = o * self.activation(memory)
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                h._uses_learning_phase = True
        return h, [h, memory]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation':
                      activations.serialize(self.recurrent_activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(
                      self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout,
                  'implementation': self.implementation}
        base_config = super(GraphLSTMCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
