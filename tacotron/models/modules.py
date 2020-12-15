import tensorflow as tf 
from tacotron.models.zoneout_LSTM import ZoneoutLSTMCell
from hparams import hparams


def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
	drop_rate = hparams.tacotron_dropout_rate

	with tf.variable_scope(scope):
		conv1d_output = tf.layers.conv1d(
			inputs,
			filters=channels,
			kernel_size=kernel_size,
			activation=None,
			padding='same')
		batched = tf.layers.batch_normalization(conv1d_output, training=is_training)
		activated = activation(batched)
		return tf.layers.dropout(activated, rate=drop_rate, training=is_training, name='dropout_{}'.format(scope))


class EncoderConvolutions:
	def __init__(self, is_training, kernel_size=(5, ), channels=512, activation=tf.nn.relu, scope=None):
		super(EncoderConvolutions, self).__init__()
		self.is_training = is_training

		self.kernel_size = kernel_size
		self.channels = channels
		self.activation = activation
		self.scope = 'enc_conv_layers' if scope is None else scope

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			x = inputs
			for i in range(3):
				x = conv1d(x, self.kernel_size, self.channels, self.activation,
					self.is_training, 'conv_layer_{}_'.format(i + 1) + self.scope)
		return x


class EncoderRNN:
	def __init__(self, is_training, size=256, zoneout=0.1, scope=None):
		super(EncoderRNN, self).__init__()
		self.is_training = is_training

		self.size = size
		self.zoneout = zoneout
		self.scope = 'encoder_LSTM' if scope is None else scope

		#Create LSTM Cell
		self._cell = ZoneoutLSTMCell(size, is_training,
			zoneout_factor_cell=zoneout,
			zoneout_factor_output=zoneout)

	def __call__(self, inputs, input_lengths):
		with tf.variable_scope(self.scope):
			outputs, (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
				self._cell,
				self._cell,
				inputs,
				sequence_length=input_lengths,
				dtype=tf.float32)

			return tf.concat(outputs, axis=2) # Concat and return forward + backward outputs


class Prenet:
	def __init__(self, is_training, layer_sizes=[256, 256], activation=tf.nn.relu, scope=None):
		super(Prenet, self).__init__()
		self.drop_rate = hparams.tacotron_dropout_rate

		self.layer_sizes = layer_sizes
		self.is_training = is_training
		self.activation = activation
		
		self.scope = 'prenet' if scope is None else scope

	def __call__(self, inputs):
		x = inputs

		with tf.variable_scope(self.scope):
			for i, size in enumerate(self.layer_sizes):
				dense = tf.layers.dense(x, units=size, activation=self.activation, name='dense_{}'.format(i + 1))

				x = tf.layers.dropout(dense, rate=self.drop_rate, training=True, name='dropout_{}'.format(i + 1) + self.scope)
		return x


class DecoderRNN:
	def __init__(self, is_training, layers=2, size=1024, zoneout=0.1, scope=None):
		super(DecoderRNN, self).__init__()
		self.is_training = is_training

		self.layers = layers
		self.size = size
		self.zoneout = zoneout
		self.scope = 'decoder_rnn' if scope is None else scope

		#Create a set of LSTM layers
		self.rnn_layers = [ZoneoutLSTMCell(size, is_training, 
			zoneout_factor_cell=zoneout,
			zoneout_factor_output=zoneout) for i in range(layers)]

		self._cell = tf.contrib.rnn.MultiRNNCell(self.rnn_layers, state_is_tuple=True)

	def __call__(self, inputs, states):
		with tf.variable_scope(self.scope):
			return self._cell(inputs, states)


class FrameProjection:
	def __init__(self, shape=80, activation=None, scope=None):
		super(FrameProjection, self).__init__()

		self.shape = shape
		self.activation = activation
		
		self.scope = 'Linear_projection' if scope is None else scope

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			output = tf.layers.dense(inputs, units=self.shape, activation=self.activation, name='projection_{}'.format(self.scope))

			return output


class StopProjection:
	def __init__(self, is_training, shape=hparams.outputs_per_step, activation=tf.nn.sigmoid, scope=None):
		super(StopProjection, self).__init__()
		self.is_training = is_training
		
		self.shape = shape
		self.activation = activation
		self.scope = 'stop_token_projection' if scope is None else scope

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			output = tf.layers.dense(inputs, units=self.shape,
				activation=None, name='projection_{}'.format(self.scope))

			#During training, don't use activation as it is integrated inside the sigmoid_cross_entropy loss function
			if self.is_training:
				return output
			return self.activation(output)


class Postnet:
	def __init__(self, is_training, kernel_size=(5, ), channels=512, activation=tf.nn.tanh, scope=None):
		super(Postnet, self).__init__()
		self.is_training = is_training

		self.kernel_size = kernel_size
		self.channels = channels
		self.activation = activation
		self.scope = 'postnet_convolutions' if scope is None else scope

	def __call__(self, inputs):
		with tf.variable_scope(self.scope):
			x = inputs
			for i in range(hparams.postnet_num_layers - 1):
				x = conv1d(x, self.kernel_size, self.channels, self.activation,
					self.is_training, 'conv_layer_{}_'.format(i + 1)+self.scope)
			x = conv1d(x, self.kernel_size, self.channels, lambda _: _, self.is_training, 'conv_layer_{}_'.format(5)+self.scope)
		return x