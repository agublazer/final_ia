import os
from tacotron.synthesizer import Synthesizer
import tensorflow as tf

sentences = ['hello the cow my friend', 'san pablo catholic university artificial intelligence', 'my name is lucifer please take my hand']


def tacotron_synthesize():
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warnings https://stackoverflow.com/questions/47068709/
	output_dir = 'A'
	checkpoint_path = tf.train.get_checkpoint_state('trained_model').model_checkpoint_path
	print('####### checkpoint_path', checkpoint_path)
	synth = Synthesizer()
	synth.load(checkpoint_path)

	os.makedirs(output_dir, exist_ok=True)

	for i, text in enumerate(sentences):
		synth.synthesize(text, i + 1, output_dir, None)

	print('Results at: {}'.format(output_dir))

tacotron_synthesize()
