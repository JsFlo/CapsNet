from __future__ import division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# puts a bad dependency - can only be run from this directory
import sys
sys.path.append('../')
# Makes them look like static method calls (not python style but helps me :)
import caps_net_model.model as CapsNetModel
import os


def create_dirs_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


MNIST = input_data.read_data_sets("/tmp/data/")

CHECKPOINT_PATH = "../checkpoint_with_decoder_tags_67_epoch/my_caps_net"
NUM_SAMPLES = 10
MASKED_OUTPUT_PATH = "./minimal_decoder_input/masked_capsules/"
SOURCE_IMAGES_OUTPUT_PATH = "./minimal_decoder_input/source_images/"
MASKED_OUT_NPY_FILENAME = "masked_output.npy"
SOURCE_IMAGES_NPY_FILENAME = "source_images.npy"

create_dirs_if_not_exists(MASKED_OUTPUT_PATH)
create_dirs_if_not_exists(SOURCE_IMAGES_OUTPUT_PATH)


def export_inputs():
    input_image_batch = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
    batch_size = tf.shape(input_image_batch)[0]

    digitCaps_postRouting, \
    decoder_output, \
    single_digit_prediction, \
    correct_labels_placeholder, masked_out = CapsNetModel.get_model_output_for_predictions(input_image_batch,
                                                                                           batch_size)

    saver = tf.train.Saver()

    source_images = MNIST.test.images[:NUM_SAMPLES].reshape([-1, 28, 28, 1])

    with tf.Session() as sess:
        saver.restore(sess, CHECKPOINT_PATH)

        digitCaps_postRouting_value, decoder_output_value, single_digit_prediction_value, masked_out_value = sess.run(
            [digitCaps_postRouting, decoder_output, single_digit_prediction, masked_out],
            feed_dict={input_image_batch: source_images,
                       correct_labels_placeholder: np.array([], dtype=np.int64)})
        # source_images = source_images.reshape(-1, 28, 28)
        # reconstructions = decoder_output_value.reshape([-1, 28, 28])

        print("shape of masked out: {}".format(masked_out_value.shape))
        np.save(MASKED_OUTPUT_PATH + MASKED_OUT_NPY_FILENAME, masked_out_value)
        np.save(SOURCE_IMAGES_OUTPUT_PATH + SOURCE_IMAGES_NPY_FILENAME, source_images)


export_inputs()