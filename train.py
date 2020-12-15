import numpy as np
from datetime import datetime
import os
import tensorflow as tf
import traceback

from tacotron.feeder import Feeder
from hparams import hparams
from tacotron.models.tacotron import Tacotron


checkpoint_interval = 100


def add_stats(model):
	with tf.variable_scope('stats') as scope:
		tf.summary.histogram('mel_outputs', model.mel_outputs)
		tf.summary.histogram('mel_targets', model.mel_targets)
		tf.summary.scalar('before_loss', model.before_loss)
		tf.summary.scalar('after_loss', model.after_loss)
		tf.summary.scalar('regularization_loss', model.regularization_loss)
		tf.summary.scalar('stop_token_loss', model.stop_token_loss)
		tf.summary.scalar('loss', model.loss)
		tf.summary.scalar('learning_rate', model.learning_rate)  # control learning rate decay speed
		gradient_norms = [tf.norm(grad) for grad in model.gradients]
		tf.summary.histogram('gradient_norm', gradient_norms)
		tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms))  # visualize gradients (in case of explosion)
		return tf.summary.merge_all()


def time_string():
	return datetime.now().strftime('%Y-%m-%d %H:%M')


def train():
	save_dir = 'trained_model_t/'
	input_path = 'training_data/train.txt'
	checkpoint_path = os.path.join(save_dir, 'model.ckpt')
	plot_dir = os.path.join(save_dir, 'plots')
	wav_dir = os.path.join(save_dir, 'wavs')
	mel_dir = os.path.join(save_dir, 'mel-spectrograms')
	os.makedirs(plot_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(mel_dir, exist_ok=True)

	# Set up data feeder
	coord = tf.train.Coordinator()
	with tf.variable_scope('datafeeder') as scope:
		feeder = Feeder(coord, input_path, hparams)

	# Set up model:
	step_count = 0

	global_step = tf.Variable(step_count, name='global_step', trainable=False)
	with tf.variable_scope('model') as scope:
		model = Tacotron(hparams)
		model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.token_targets)
		model.add_loss()
		model.add_optimizer(global_step)
		stats = add_stats(model)

	step = 0
	saver = tf.train.Saver(max_to_keep=5)

	config = tf.ConfigProto()

	# Train
	with tf.Session(config=config) as sess:
		try:
			sess.run(tf.global_variables_initializer())

			feeder.start_in_session(sess)

			# Training loop
			while not coord.should_stop():
				step, loss, opt = sess.run([global_step, model.loss, model.optimize])
				message = 'Step {:7d} , loss={:.5f}'.format(step, loss)
				print(message)

				if step % 100 == 0:
					with open(os.path.join(save_dir, 'step_counter.txt'), 'w') as file:
						file.write(str(step))
					print('Saving checkpoint to: {}-{}'.format(checkpoint_path, step))
					saver.save(sess, checkpoint_path, global_step=step)

		except Exception as e:
			print('Exception: {}'.format(e))
			traceback.print_exc()
			coord.request_stop(e)

train()
