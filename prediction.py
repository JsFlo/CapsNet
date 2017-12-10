from __future__ import division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Makes them look like static method calls (not python style but helps me :)
import caps_net_model.model as CapsNetModel

MNIST = input_data.read_data_sets("/tmp/data/")

# CHECKPOINT_PATH = "./checkpoints/my_caps_net"
CHECKPOINT_PATH = "./checkpoint_with_decoder_tags_67_epoch/my_caps_net"
NUM_SAMPLES = 1


def predict():
    input_image_batch = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
    batch_size = tf.shape(input_image_batch)[0]

    digitCaps_postRouting, \
    decoder_output, \
    single_digit_prediction, \
    correct_labels_placeholder, masked_out = CapsNetModel.get_model_output_for_predictions(input_image_batch,
                                                                                           batch_size)

    saver = tf.train.Saver()

    sample_images = MNIST.test.images[:NUM_SAMPLES].reshape([-1, 28, 28, 1])

    with tf.Session() as sess:
        saver.restore(sess, CHECKPOINT_PATH)

        digitCaps_postRouting_value, decoder_output_value, single_digit_prediction_value, masked_out_value = sess.run(
            [digitCaps_postRouting, decoder_output, single_digit_prediction, masked_out],
            feed_dict={input_image_batch: sample_images,
                       correct_labels_placeholder: np.array([], dtype=np.int64)})
        sample_images = sample_images.reshape(-1, 28, 28)
        reconstructions = decoder_output_value.reshape([-1, 28, 28])

        print("shape of masked out: {}".format(masked_out_value[0]))
        np.save('example_input_for_decoder_67/example_decoder_67_input.npy', masked_out_value[0])
        plt.figure(figsize=(300, 7))
        for index in range(NUM_SAMPLES):
            plt.subplot(2, NUM_SAMPLES, index + 1)
            plt.imshow(sample_images[index], cmap="binary")
            plt.axis("off")
            plt.title("Label:" + str(MNIST.test.labels[index]))

            plt.subplot(2, NUM_SAMPLES, index + NUM_SAMPLES + 1)
            plt.imshow(reconstructions[index], cmap="binary")
            plt.axis("off")
            plt.title("Predicted:" + str(single_digit_prediction_value[index]))

        plt.show()


predict()
