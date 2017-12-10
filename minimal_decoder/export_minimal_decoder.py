from __future__ import division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
# puts a bad dependency - can only be run from this directory
import sys

sys.path.append('../')

# Makes them look like static method calls (not python style but helps me :)
import caps_net_model.model as CapsNetModel
import argparse
from tensorflow.examples.tutorials.mnist import input_data

MNIST = input_data.read_data_sets("/tmp/data/")

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True,
                    help='Path to CHECKPOINT_NAME file (without ext) ([CHECKPOINT_NAME].meta, [CHECKPOINT_NAME].index, etc).')
parser.add_argument('--model_output', type=str, required=False, default='./model_output')
parser.add_argument('--model_name', type=str, required=False, default='model_graph_v2.pb')
parser.add_argument('--masked_capsules', type=int, required=False, default=30)
FLAGS = parser.parse_args()

MASKED_OUTPUT_PATH = FLAGS.model_output + "/masked_capsules/"
MASKED_OUT_NPY_FILENAME = "masked_output.npy"

SOURCE_IMAGES_OUTPUT_PATH = FLAGS.model_output + "/source_images/"
SOURCE_IMAGES_NPY_FILENAME = "source_images.npy"


# CHECKPOINT_PATH = "../checkpoint_with_decoder_tags_67_epoch/my_caps_net"
# EXPORT_DIR = './model_output'
# MODEL_NAME = 'decoder_67_v2.pb'
def create_dirs_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


create_dirs_if_not_exists(FLAGS.model_output)
create_dirs_if_not_exists(MASKED_OUTPUT_PATH)
create_dirs_if_not_exists(SOURCE_IMAGES_OUTPUT_PATH)


def exportGraph(g, W1, B1, W2, B2, W3, B3):
    with g.as_default():
        masked_input = tf.placeholder(dtype="float32", shape=[None, 1, 10, 16, 1], name="input")
        decoder_input = tf.reshape(masked_input, [-1, 10 * 16])

        WC1 = tf.constant(W1, name="WC1")
        BC1 = tf.constant(B1, name="BC1")

        hidden1 = tf.nn.relu(tf.matmul(decoder_input, WC1) + BC1)

        WC2 = tf.constant(W2, name="WC2")
        BC2 = tf.constant(B2, name="BC2")
        hidden2 = tf.nn.relu(tf.matmul(hidden1, WC2) + BC2)
        WC3 = tf.constant(W3, name="WC3")
        BC3 = tf.constant(B3, name="BC3")
        decoder_output = tf.nn.sigmoid(tf.matmul(hidden2, WC3) + BC3, name="output")
        #
        # hidden1 = tf.layers.dense(decoder_input, 512, kernel_initializer=WC1, bias_initializer=BC1,
        #                           activation=tf.nn.relu)
        # hidden2 = tf.layers.dense(hidden1, 1024, kernel_initializer=WC2, bias_initializer=BC2,
        #                           activation=tf.nn.relu)
        # decoder_output = tf.layers.dense(hidden2, 28 * 28, kernel_initializer=WC3, bias_initializer=BC3,
        #                                  activation=tf.nn.sigmoid)
        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        graph_def = g.as_graph_def()
        tf.train.write_graph(graph_def, FLAGS.model_output, FLAGS.model_name, as_text=False)


def restore_export():
    input_image_batch = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
    batch_size = tf.shape(input_image_batch)[0]

    digitCaps_postRouting, \
    decoder_output, \
    single_digit_prediction, \
    mask_with_labels, \
    correct_labels_placeholder, masked_out = CapsNetModel.get_model_output_for_tweak(input_image_batch, batch_size)

    all_var = tf.trainable_variables()
    decoder_weights = [var for var in all_var if 'decoder' in var.name]
    print("Decoder Weights")
    for weight_tensor in decoder_weights:
        print("Tensors shapes: {}".format(weight_tensor.shape))

    saver = tf.train.Saver()
    source_images = MNIST.test.images[:FLAGS.masked_capsules].reshape([-1, 28, 28, 1])

    with tf.Session() as sess:
        # restore
        saver.restore(sess, FLAGS.checkpoint)

        # save minimal decoder input
        digitCaps_postRouting_value, decoder_output_value, single_digit_prediction_value, masked_out_value = sess.run(
            [digitCaps_postRouting, decoder_output, single_digit_prediction, masked_out],
            feed_dict={input_image_batch: source_images,
                       correct_labels_placeholder: np.array([], dtype=np.int64)})

        print("Masked capsules shape: {}".format(masked_out_value.shape))
        np.save(MASKED_OUTPUT_PATH + MASKED_OUT_NPY_FILENAME, masked_out_value)
        np.save(SOURCE_IMAGES_OUTPUT_PATH + SOURCE_IMAGES_NPY_FILENAME, source_images)

        # save out small model
        W1 = decoder_weights[0].eval(sess)
        B1 = decoder_weights[1].eval(sess)

        W2 = decoder_weights[2].eval(sess)
        B2 = decoder_weights[3].eval(sess)

        W3 = decoder_weights[4].eval(sess)
        B3 = decoder_weights[5].eval(sess)

    # Create new graph for exporting
    g = tf.Graph()
    exportGraph(g, W1, B1, W2, B2, W3, B3)


restore_export()
