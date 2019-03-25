from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.engine import Layer, InputSpec
from keras.layers import RNN, has_arg, to_list, np
from keras.layers.recurrent import _standardize_args
import tensorflow as tf
from tensorflow import expand_dims, reverse, zeros_like
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


class GraphLSTM(RNN):
    def __init__(self, cell,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if not hasattr(cell, 'call'):
            raise ValueError('`cell` should have a `call` method. '
                             'The RNN was passed:', cell)
        if not hasattr(cell, 'state_size'):
            raise ValueError('The RNN cell should have '
                             'an attribute `state_size` '
                             '(tuple of integers, '
                             'one integer per RNN state).')
        super(RNN, self).__init__(**kwargs)
        self.cell = cell
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.stateful = stateful
        self.unroll = unroll

        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        self.state_spec = None
        self._states = None
        self.constants_spec = None
        self._num_constants = 4

    @property
    def states(self):
        if self._states is None:
            if isinstance(self.cell.state_size, int):
                num_states = 1
            else:
                num_states = len(self.cell.state_size)
            return [None for _ in range(num_states)]
        return self._states

    @states.setter
    def states(self, states):
        self._states = states

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        if hasattr(self.cell.state_size, '__len__'):
            state_size = self.cell.state_size
        else:
            state_size = [self.cell.state_size]

        if getattr(self.cell, 'output_size', None) is not None:
            output_dim = self.cell.output_size
        else:
            output_dim = state_size[0]

        if self.return_sequences:
            output_shape = (input_shape[0], input_shape[1], output_dim)
        else:
            output_shape = (input_shape[0], output_dim)

        if self.return_state:
            state_shape = [(input_shape[0], dim) for dim in state_size]
            return [output_shape] + state_shape
        else:
            return output_shape

    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            mask = mask[0]
        output_mask = mask if self.return_sequences else None
        if self.return_state:
            state_mask = [None for _ in self.states]
            return [output_mask] + state_mask
        else:
            return output_mask

    def build(self, input_shape):
        # Note input_shape will be list of shapes of initial states and
        # constants if these are passed in __call__.
        if self._num_constants is not None:
            constants_shape = input_shape[-self._num_constants:]
        else:
            constants_shape = None

        if isinstance(input_shape, list):
            input_shape = input_shape[0]

        batch_size = input_shape[0] if self.stateful else None
        input_dim = input_shape[-1]
        self.input_spec[0] = InputSpec(shape=(batch_size, None, input_dim))

        # allow cell (if layer) to build before we set or validate state_spec
        if isinstance(self.cell, Layer):
            step_input_shape = (input_shape[0],) + input_shape[2:]
            if constants_shape is not None:
                self.cell.build([step_input_shape] + constants_shape)
            else:
                self.cell.build(step_input_shape)

        # set or validate state_spec
        if hasattr(self.cell.state_size, '__len__'):
            state_size = list(self.cell.state_size)
        else:
            state_size = [self.cell.state_size]

        if self.state_spec is not None:
            # initial_state was passed in call, check compatibility
            if [spec.shape[-1] for spec in self.state_spec] != state_size:
                raise ValueError(
                    'An `initial_state` was passed that is not compatible with '
                    '`cell.state_size`. Received `state_spec`={}; '
                    'however `cell.state_size` is '
                    '{}'.format(self.state_spec, self.cell.state_size))
        else:
            self.state_spec = [InputSpec(shape=(None, dim))
                               for dim in state_size]
        if self.stateful:
            self.reset_states()
        self.built = True

    def get_initial_state(self, inputs):
        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)
        if hasattr(self.cell.state_size, '__len__'):
            return [K.tile(initial_state, [1, dim])
                    for dim in self.cell.state_size]
        else:
            return [K.tile(initial_state, [1, self.cell.state_size])]

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        inputs, initial_state, constants = _standardize_args(
            inputs, initial_state, constants, self._num_constants)

        if initial_state is None and constants is None:
            return super(RNN, self).__call__(inputs, **kwargs)

        # If any of `initial_state` or `constants` are specified and are Keras
        # tensors, then add them to the inputs and temporarily modify the
        # input_spec to include them.

        additional_inputs = []
        additional_specs = []
        if initial_state is not None:
            kwargs['initial_state'] = initial_state
            additional_inputs += initial_state
            self.state_spec = [InputSpec(shape=K.int_shape(state))
                               for state in initial_state]
            additional_specs += self.state_spec
        if constants is not None:
            kwargs['constants'] = constants
            additional_inputs += constants
            self.constants_spec = [InputSpec(shape=K.int_shape(constant))
                                   for constant in constants]
            self._num_constants = len(constants)
            additional_specs += self.constants_spec
        # at this point additional_inputs cannot be empty
        is_keras_tensor = K.is_keras_tensor(additional_inputs[0])
        for tensor in additional_inputs:
            if K.is_keras_tensor(tensor) != is_keras_tensor:
                raise ValueError('The initial state or constants of an RNN'
                                 ' layer cannot be specified with a mix of'
                                 ' Keras tensors and non-Keras tensors'
                                 ' (a "Keras tensor" is a tensor that was'
                                 ' returned by a Keras layer, or by `Input`)')

        if is_keras_tensor:
            # Compute the full input spec, including state and constants
            full_input = [inputs] + additional_inputs
            full_input_spec = self.input_spec + additional_specs
            # Perform the call with temporarily replaced input_spec
            original_input_spec = self.input_spec
            self.input_spec = full_input_spec
            output = super(RNN, self).__call__(full_input, **kwargs)
            self.input_spec = original_input_spec
            return output
        else:
            return super(RNN, self).__call__(inputs, **kwargs)

    def call(self,
             inputs,
             mask=None,
             training=None,
             initial_state=None,
             constants=None):
        # input shape: `(samples, time (padded with zeros), input_dim)`
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            # get initial_state from full input spec
            # as they could be copied to multiple GPU.
            if self._num_constants is None:
                initial_state = inputs[1:]
            else:
                initial_state = inputs[1:-self._num_constants]
            if len(initial_state) == 0:
                initial_state = None
            inputs = inputs[0]
        if initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)

        if isinstance(mask, list):
            mask = mask[0]

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')
        input_shape = K.int_shape(inputs)
        timesteps = input_shape[1]
        if self.unroll and timesteps in [None, 1]:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined or equal to 1. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')

        kwargs = {}
        kwargs['previous_state'] = initial_state
        if has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        if constants:
            if not has_arg(self.cell.call, 'constants'):
                raise ValueError('RNN cell does not support constants')

            def step(inputs, states):
                constants = states[-self._num_constants:]
                states = states[:-self._num_constants]
                return self.cell.call(inputs, states, constants=constants,
                                      **kwargs)
        else:
            def step(inputs, states):
                return self.cell.call(inputs, states, **kwargs)

        last_output, outputs, states = self.rnn(step,
                                                inputs,
                                                initial_state,
                                                constants=constants,
                                                go_backwards=self.go_backwards,
                                                mask=mask,
                                                unroll=self.unroll,
                                                input_length=timesteps)
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        if self.return_sequences:
            output = outputs
        else:
            output = last_output

        # Properly set learning phase
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True
            for state in states:
                state._uses_learning_phase = True

        if self.return_state:
            states = to_list(states, allow_tuple=True)
            return [output] + states
        else:
            return output

    def reset_states(self, states=None):
        if not self.stateful:
            raise AttributeError('Layer must be stateful.')
        batch_size = self.input_spec[0].shape[0]
        if not batch_size:
            raise ValueError('If a RNN is stateful, it needs to know '
                             'its batch size. Specify the batch size '
                             'of your input tensors: \n'
                             '- If using a Sequential model, '
                             'specify the batch size by passing '
                             'a `batch_input_shape` '
                             'argument to your first layer.\n'
                             '- If using the functional API, specify '
                             'the batch size by passing a '
                             '`batch_shape` argument to your Input layer.')
        # initialize state if None
        if self.states[0] is None:
            if hasattr(self.cell.state_size, '__len__'):
                self.states = [K.zeros((batch_size, dim))
                               for dim in self.cell.state_size]
            else:
                self.states = [K.zeros((batch_size, self.cell.state_size))]
        elif states is None:
            if hasattr(self.cell.state_size, '__len__'):
                for state, dim in zip(self.states, self.cell.state_size):
                    K.set_value(state, np.zeros((batch_size, dim)))
            else:
                K.set_value(self.states[0],
                            np.zeros((batch_size, self.cell.state_size)))
        else:
            states = to_list(states, allow_tuple=True)
            if len(states) != len(self.states):
                raise ValueError('Layer ' + self.name + ' expects ' +
                                 str(len(self.states)) + ' states, '
                                                         'but it received ' + str(
                    len(states)) +
                                 ' state values. Input received: ' +
                                 str(states))
            for index, (value, state) in enumerate(zip(states, self.states)):
                if hasattr(self.cell.state_size, '__len__'):
                    dim = self.cell.state_size[index]
                else:
                    dim = self.cell.state_size
                if value.shape != (batch_size, dim):
                    raise ValueError('State ' + str(index) +
                                     ' is incompatible with layer ' +
                                     self.name + ': expected shape=' +
                                     str((batch_size, dim)) +
                                     ', found shape=' + str(value.shape))
                # TODO: consider batch calls to `set_value`.
                K.set_value(state, value)

    def get_config(self):
        config = {'return_sequences': self.return_sequences,
                  'return_state': self.return_state,
                  'go_backwards': self.go_backwards,
                  'stateful': self.stateful,
                  'unroll': self.unroll}
        if self._num_constants is not None:
            config['num_constants'] = self._num_constants

        cell_config = self.cell.get_config()
        config['cell'] = {'class_name': self.cell.__class__.__name__,
                          'config': cell_config}
        base_config = super(RNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from keras.layers import deserialize as deserialize_layer
        cell = deserialize_layer(config.pop('cell'),
                                 custom_objects=custom_objects)
        num_constants = config.pop('num_constants', None)
        layer = cls(cell, **config)
        layer._num_constants = num_constants
        return layer

    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        if isinstance(self.cell, Layer):
            return self.cell.trainable_weights
        return []

    @property
    def non_trainable_weights(self):
        if isinstance(self.cell, Layer):
            if not self.trainable:
                return self.cell.weights
            return self.cell.non_trainable_weights
        return []

    @property
    def losses(self):
        layer_losses = super(RNN, self).losses
        if isinstance(self.cell, Layer):
            return self.cell.losses + layer_losses
        return layer_losses

    def get_losses_for(self, inputs=None):
        if isinstance(self.cell, Layer):
            cell_losses = self.cell.get_losses_for(inputs)
            return cell_losses + super(RNN, self).get_losses_for(inputs)
        return super(RNN, self).get_losses_for(inputs)

    def rnn(self, step_function, inputs, initial_states,
            go_backwards=False, mask=None, constants=None,
            unroll=False, input_length=None):
        ndim = len(inputs.get_shape())
        if ndim < 3:
            raise ValueError('Input should be at least 3D.')

        # Transpose to time-major, i.e.
        # from (batch, time, ...) to (time, batch, ...)
        axes = [1, 0] + list(range(2, ndim))
        inputs = tf.transpose(inputs, (axes))

        if mask is not None:
            if mask.dtype != tf.bool:
                mask = tf.cast(mask, tf.bool)
            if len(mask.get_shape()) == ndim - 1:
                mask = expand_dims(mask)
            mask = tf.transpose(mask, axes)

        if constants is None:
            constants = []

        global uses_learning_phase
        uses_learning_phase = False

        if unroll:
            if not inputs.get_shape()[0]:
                raise ValueError('Unrolling requires a '
                                 'fixed number of timesteps.')
            states = initial_states
            successive_states = []
            successive_outputs = []

            input_list = tf.unstack(inputs)
            if go_backwards:
                input_list.reverse()

            if mask is not None:
                mask_list = tf.unstack(mask)
                if go_backwards:
                    mask_list.reverse()

                for inp, mask_t in zip(input_list, mask_list):
                    output, new_states = step_function(inp, states + constants)
                    if getattr(output, '_uses_learning_phase', False):
                        uses_learning_phase = True

                    # tf.where needs its condition tensor
                    # to be the same shape as its two
                    # result tensors, but in our case
                    # the condition (mask) tensor is
                    # (nsamples, 1), and A and B are (nsamples, ndimensions).
                    # So we need to
                    # broadcast the mask to match the shape of A and B.
                    # That's what the tile call does,
                    # it just repeats the mask along its second dimension
                    # n times.
                    tiled_mask_t = tf.tile(mask_t,
                                           tf.stack([1, tf.shape(output)[1]]))

                    if not successive_outputs:
                        prev_output = zeros_like(output)
                    else:
                        prev_output = successive_outputs[-1]

                    output = tf.where(tiled_mask_t, output, prev_output)

                    return_states = []
                    for state, new_state in zip(states, new_states):
                        # (see earlier comment for tile explanation)
                        tiled_mask_t = tf.tile(mask_t,
                                               tf.stack([1,
                                                         tf.shape(new_state)[
                                                             1]]))
                        return_states.append(tf.where(tiled_mask_t,
                                                      new_state,
                                                      state))
                    states = return_states
                    successive_outputs.append(output)
                    successive_states.append(states)
                last_output = successive_outputs[-1]
                new_states = successive_states[-1]
                outputs = tf.stack(successive_outputs)
            else:
                for inp in input_list:
                    output, states = step_function(inp, states + constants)
                    if getattr(output, '_uses_learning_phase', False):
                        uses_learning_phase = True
                    successive_outputs.append(output)
                    successive_states.append(states)
                last_output = successive_outputs[-1]
                new_states = successive_states[-1]
                outputs = tf.stack(successive_outputs)

        else:
            if go_backwards:
                inputs = reverse(inputs, 0)

            states = tuple(initial_states)

            time_steps = tf.shape(inputs)[0]
            outputs, _ = step_function(inputs[0], initial_states + constants)
            output_ta = tensor_array_ops.TensorArray(
                dtype=outputs.dtype,
                size=time_steps,
                tensor_array_name='output_ta')
            input_ta = tensor_array_ops.TensorArray(
                dtype=inputs.dtype,
                size=time_steps,
                tensor_array_name='input_ta')
            input_ta = input_ta.unstack(inputs)
            time = tf.constant(0, dtype='int32', name='time')

            if mask is not None:
                if not states:
                    raise ValueError('No initial states provided! '
                                     'When using masking in an RNN, you should '
                                     'provide initial states '
                                     '(and your step function should return '
                                     'as its first state at time `t` '
                                     'the output at time `t-1`).')
                if go_backwards:
                    mask = reverse(mask, 0)

                mask_ta = tensor_array_ops.TensorArray(
                    dtype=tf.bool,
                    size=time_steps,
                    tensor_array_name='mask_ta')
                mask_ta = mask_ta.unstack(mask)

                def _step(time, output_ta_t, *states):
                    """RNN step function.

                    # Arguments
                        time: Current timestep value.
                        output_ta_t: TensorArray.
                        *states: List of states.

                    # Returns
                        Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
                    """
                    current_input = input_ta.read(time)
                    mask_t = mask_ta.read(time)
                    output, new_states = step_function(current_input,
                                                       tuple(states) +
                                                       tuple(constants))
                    if getattr(output, '_uses_learning_phase', False):
                        global uses_learning_phase
                        uses_learning_phase = True
                    for state, new_state in zip(states, new_states):
                        new_state.set_shape(state.get_shape())
                    tiled_mask_t = tf.tile(mask_t,
                                           tf.stack([1, tf.shape(output)[1]]))
                    output = tf.where(tiled_mask_t, output, states[0])
                    new_states = [
                        tf.where(tf.tile(mask_t, tf.stack(
                            [1, tf.shape(new_states[i])[1]])),
                                 new_states[i], states[i]) for i in
                        range(len(states))
                    ]
                    output_ta_t = output_ta_t.write(time, output)
                    return (time + 1, output_ta_t) + tuple(new_states)
            else:
                def _step(time, output_ta_t, *states):
                    """RNN step function.

                    # Arguments
                        time: Current timestep value.
                        output_ta_t: TensorArray.
                        *states: List of states.

                    # Returns
                        Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
                    """
                    current_input = input_ta.read(time)
                    output, new_states = step_function(current_input,
                                                       tuple(states) +
                                                       tuple(constants))
                    if getattr(output, '_uses_learning_phase', False):
                        global uses_learning_phase
                        uses_learning_phase = True
                    for state, new_state in zip(states, new_states):
                        new_state.set_shape(state.get_shape())
                    output_ta_t = output_ta_t.write(time, output)
                    return (time + 1, output_ta_t) + tuple(new_states)

            final_outputs = control_flow_ops.while_loop(
                cond=lambda time, *_: time < time_steps,
                body=_step,
                loop_vars=(time, output_ta) + states,
                parallel_iterations=32,
                swap_memory=True,
                maximum_iterations=input_length)
            last_time = final_outputs[0]
            output_ta = final_outputs[1]
            new_states = final_outputs[2:]

            outputs = output_ta.stack()
            last_output = output_ta.read(last_time - 1)

        axes = [1, 0] + list(range(2, len(outputs.get_shape())))
        outputs = tf.transpose(outputs, axes)
        last_output._uses_learning_phase = uses_learning_phase
        return last_output, outputs, new_states
