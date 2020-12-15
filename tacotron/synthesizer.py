import os
import numpy as np
import tensorflow as tf
from hparams import hparams
# from tacotron.models import create_model
from tacotron.utils.text import text_to_sequence
from tacotron.utils import plot
from datasets import audio
from tacotron.models.tacotron import Tacotron


class Synthesizer:
	def load(self, checkpoint_path):
		inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
		input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')

		with tf.variable_scope('model') as scope:  # no se puede quitar, entender
			# self.model = create_model(hparams)
			self.model = Tacotron(hparams)
			self.model.initialize(inputs, input_lengths)
			self.mel_outputs = self.model.mel_outputs
			self.alignment = self.model.alignments[0]

		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		saver.restore(self.session, checkpoint_path)

	def synthesize(self, text, idx, out_dir, mel_filename):
		cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		seq = text_to_sequence(text, cleaner_names)
		feed_dict = {
			self.model.inputs: [np.asarray(seq, dtype=np.int32)],
			self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32),
		}

		mels, alignment = self.session.run([self.mel_outputs, self.alignment], feed_dict=feed_dict)

		mels = mels.reshape(-1, hparams.num_mels)

		wav = audio.inv_mel_spectrogram(mels.T)
		audio.save_wav(wav, os.path.join(out_dir, 'audio-{:02d}.wav'.format(idx)))

		# save mel spectrogram plot
		plot.plot_spectrogram(mels, os.path.join(out_dir, 'mel-{:02d}.png'.format(idx)),
				info='{}'.format(text), split_title=True)

		return 1
