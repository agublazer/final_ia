import tensorflow as tf
from tensorflow.python.ops.rnn_cell import RNNCell


class ZoneoutLSTMCell(RNNCell):
    """Zoneout Regularization for LSTM-RNN.
    """

    def __init__(self, num_units, is_training,
         initializer=tf.contrib.layers.xavier_initializer(),
         forget_bias=1.0, state_is_tuple=True, activation=tf.tanh, zoneout_factor_cell=0.0, zoneout_factor_output=0.0,
         reuse=None):
        """Initialize the parameters for an LSTM cell.
        Args:
          num_units: int, The number of units in the LSTM cell.
          is_training: bool, set True when training.
          forget_bias: Biases of the forget gate are initialized by default
            to 1 in order to reduce the scale of forgetting at the beginning of
            the training.
          activation: Activation function of the inner states.
        """
        self.num_units = num_units
        self.is_training = is_training
        self.initializer = initializer
        self.forget_bias = forget_bias
        self.state_is_tuple = state_is_tuple
        self.activation = activation
        self.zoneout_factor_cell = zoneout_factor_cell
        self.zoneout_factor_output = zoneout_factor_output

        self._state_size = tf.nn.rnn_cell.LSTMStateTuple(num_units, num_units)
        self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        (c_prev, h_prev) = state

        # c_prev : Tensor with the size of [batch_size, state_size]
        # h_prev : Tensor with the size of [batch_size, state_size/2]

        with tf.variable_scope(scope or type(self).__name__):

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            lstm_matrix = _linear([inputs, h_prev], 4 * self.num_units, True)
            i, j, f, o = tf.split(lstm_matrix, 4, 1)

            with tf.name_scope(None, "zoneout"):
                # make binary mask tensor for cell
                keep_prob_cell = tf.convert_to_tensor(
                    self.zoneout_factor_cell,
                    dtype=c_prev.dtype
                )
                random_tensor_cell = keep_prob_cell
                random_tensor_cell += tf.random_uniform(tf.shape(c_prev), seed=None, dtype=c_prev.dtype)
                binary_mask_cell = tf.floor(random_tensor_cell)
                # 0 <-> 1 swap
                binary_mask_cell_complement = tf.ones(tf.shape(c_prev)) - binary_mask_cell

                # make binary mask tensor for output
                keep_prob_output = tf.convert_to_tensor(
                    self.zoneout_factor_output,
                    dtype=h_prev.dtype
                )
                random_tensor_output = keep_prob_output
                random_tensor_output += tf.random_uniform(tf.shape(h_prev), seed=None, dtype=h_prev.dtype)
                binary_mask_output = tf.floor(random_tensor_output)
                # 0 <-> 1 swap
                binary_mask_output_complement = tf.ones(tf.shape(h_prev)) - binary_mask_output

            c_temp = c_prev * tf.sigmoid(f + self.forget_bias) + tf.sigmoid(i) * self.activation(j)
            if self.is_training and self.zoneout_factor_cell > 0.0:
                c = binary_mask_cell * c_prev + binary_mask_cell_complement * c_temp
            else:
                c = c_temp

            h_temp = tf.sigmoid(o) * self.activation(c)
            if self.is_training and self.zoneout_factor_output > 0.0:
                h = binary_mask_output * h_prev + binary_mask_output_complement * h_temp
            else:
                h = h_temp

            new_state = (tf.nn.rnn_cell.LSTMStateTuple(c, h)
                         if self.state_is_tuple else tf.concat(1, [c, h]))

            return h, new_state


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(args, 1), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
    return res + bias_term
