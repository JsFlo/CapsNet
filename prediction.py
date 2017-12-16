from __future__ import division, print_function, unicode_literals
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import argparse
from tensorflow.examples.tutorials.mnist import input_data

# Makes them look like static method calls (not python style but helps me :)
import caps_net_model.model as CapsNetModel

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--n_samples', type=int, default=1)
FLAGS = parser.parse_args()

MNIST = input_data.read_data_sets("/tmp/data/")


def predict():
    input_image_batch = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
    batch_size = tf.shape(input_image_batch)[0]

    digitCaps_postRouting, \
    decoder_output, \
    single_digit_prediction, \
    correct_labels_placeholder, masked_out = CapsNetModel.get_model_output_for_predictions(input_image_batch,
                                                                                           batch_size)

    saver = tf.train.Saver()

    sample_images = MNIST.test.images[:FLAGS.n_samples].reshape([-1, 28, 28, 1])

    with tf.Session() as sess:
        saver.restore(sess, FLAGS.checkpoint)

        digitCaps_postRouting_value, decoder_output_value, single_digit_prediction_value, masked_out_value = sess.run(
            [digitCaps_postRouting, decoder_output, single_digit_prediction, masked_out],
            feed_dict={input_image_batch: sample_images,
                       correct_labels_placeholder: np.array([], dtype=np.int64)})
        sample_images = sample_images.reshape(-1, 28, 28)
        reconstructions = decoder_output_value.reshape([-1, 28, 28])

        plt.figure(figsize=(300, 7))
        for index in range(FLAGS.n_samples):
            plt.subplot(2, FLAGS.n_samples, index + 1)
            plt.imshow(sample_images[index], cmap="binary")
            plt.axis("off")
            plt.title("Label:" + str(MNIST.test.labels[index]))

            plt.subplot(2, FLAGS.n_samples, index + FLAGS.n_samples + 1)
            plt.imshow(reconstructions[index], cmap="binary")
            plt.axis("off")
            plt.title("Predicted:" + str(single_digit_prediction_value[index]))

        plt.show()


predict()
