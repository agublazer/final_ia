import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import BahdanauAttention
from tensorflow.python.ops import nn_ops
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import math_ops
from hparams import hparams


def _compute_attention(attention_mechanism, cell_output, attention_state,
					   attention_layer):

	alignments, next_attention_state = attention_mechanism(cell_output, state=attention_state)

	expanded_alignments = array_ops.expand_dims(alignments, 1)
	context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
	context = array_ops.squeeze(context, [1])

	if attention_layer is not None:
		attention = attention_layer(array_ops.concat([cell_output, context], 1))
	else:
		attention = context

	return attention, alignments, next_attention_state


def _location_sensitive_score(W_query, W_fil, W_keys):
	"""Impelements Bahdanau-style (cumulative) scoring function.
	This attention is described in:
		J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
	  gio, “Attention-based models for speech recognition,” in Ad-
	  vances in Neural Information Processing Systems, 2015, pp.
	  577–585.

	#############################################################################
			  hybrid attention (content-based + location-based)
							   f = F * α_{i-1}
	   energy = dot(v_a, tanh(W_keys(h_enc) + W_query(h_dec) + W_fil(f) + b_a))
	#############################################################################

	Args:
		W_query: Tensor, shape '[batch_size, 1, attention_dim]' to compare to location features.
		W_location: processed previous alignments into location features, shape '[batch_size, max_time, attention_dim]'
		W_keys: Tensor, shape '[batch_size, max_time, attention_dim]', typically the encoder outputs.
	Returns:
		A '[batch_size, max_time]' attention score (energy)
	"""
	# Get the number of hidden units from the trailing dimension of keys
	dtype = W_query.dtype
	num_units = W_keys.shape[-1].value or array_ops.shape(W_keys)[-1]

	v_a = tf.get_variable(
		'attention_variable', shape=[num_units], dtype=dtype,
		initializer=tf.contrib.layers.xavier_initializer())
	b_a = tf.get_variable(
		'attention_bias', shape=[num_units], dtype=dtype,
		initializer=tf.zeros_initializer())

	return tf.reduce_sum(v_a * tf.tanh(W_keys + W_query + W_fil + b_a), [2])

def _smoothing_normalization(e):
	"""Applies a smoothing normalization function instead of softmax
	Introduced in:
		J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
	  gio, “Attention-based models for speech recognition,” in Ad-
	  vances in Neural Information Processing Systems, 2015, pp.
	  577–585.

	############################################################################
						Smoothing normalization function
				a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
	############################################################################

	Args:
		e: matrix [batch_size, max_time(memory_time)]: expected to be energy (score)
			values of an attention mechanism
	Returns:
		matrix [batch_size, max_time]: [0, 1] normalized alignments with possible
			attendance to multiple memory time steps.
	"""
	return tf.nn.sigmoid(e) / tf.reduce_sum(tf.nn.sigmoid(e), axis=-1, keepdims=True)


class LocationSensitiveAttention(BahdanauAttention):
	# Attention-based models for speech recognition

	def __init__(self, num_units, memory, mask_encoder=True, memory_sequence_length=None, smoothing=False,
				 cumulate_weights=True, name='LocationSensitiveAttention'):

		normalization_function = _smoothing_normalization if (smoothing == True) else None
		memory_length = memory_sequence_length if (mask_encoder==True) else None
		super(LocationSensitiveAttention, self).__init__(
				num_units=num_units,
				memory=memory,
				memory_sequence_length=memory_length,
				probability_fn=normalization_function,
				name=name)

		self.location_convolution = tf.layers.Conv1D(filters=hparams.attention_filters,
			kernel_size=hparams.attention_kernel, padding='same', use_bias=False, 
			name='location_features_convolution')
		self.location_layer = tf.layers.Dense(units=num_units, use_bias=False, 
			dtype=tf.float32, name='location_features_layer')
		self._cumulate = cumulate_weights

	def __call__(self, query, state):
		"""Score the query based on the keys and values.
		Args:
			query: Tensor of dtype matching `self.values` and shape
				`[batch_size, query_depth]`.
			state (previous alignments): Tensor of dtype matching `self.values` and shape
				`[batch_size, alignments_size]`
				(`alignments_size` is memory's `max_time`).
		Returns:
			alignments: Tensor of dtype matching `self.values` and shape
				`[batch_size, alignments_size]` (`alignments_size` is memory's
				`max_time`).
		"""
		previous_alignments = state
		with variable_scope.variable_scope(None, "Location_Sensitive_Attention", [query]):

			# processed_query shape [batch_size, query_depth] -> [batch_size, attention_dim]
			processed_query = self.query_layer(query) if self.query_layer else query
			# -> [batch_size, 1, attention_dim]
			processed_query = tf.expand_dims(processed_query, 1)

			# processed_location_features shape [batch_size, max_time, attention dimension]
			# [batch_size, max_time] -> [batch_size, max_time, 1]
			expanded_alignments = tf.expand_dims(previous_alignments, axis=2)
			# location features [batch_size, max_time, filters]
			f = self.location_convolution(expanded_alignments)
			# Projected location features [batch_size, max_time, attention_dim]
			processed_location_features = self.location_layer(f)

			# energy shape [batch_size, max_time]
			energy = _location_sensitive_score(processed_query, processed_location_features, self.keys)

		# alignments shape = energy shape = [batch_size, max_time]
		alignments = self._probability_fn(energy, previous_alignments)

		# Cumulate alignments
		if self._cumulate:
			next_state = alignments + previous_alignments
		else:
			next_state = alignments
			
		return alignments, next_state
