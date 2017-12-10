from __future__ import division, print_function, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Makes them look like static method calls (not python style but helps me :)
import caps_net_model.model as CapsNetModel

MNIST = input_data.read_data_sets("/tmp/data/")

CHECKPOINT_PATH = "./checkpoints/my_caps_net"
NUM_SAMPLES = 3
CAPS2_N_DIMS = 16
CAPS2_N_CAPS = 10
EXPORT_DIR = './minimal_decoder'
MODEL_NAME = 'base_try.pb'


def exportGraph(g, W1, B1, W2, B2, W3, B3):
    with g.as_default():
        # ", shape=(?, 1, 10, 16, 1), dtype=float32)
        masked_input = tf.placeholder(dtype="float32", shape=[None, 1, 10, 16, 1])
        decoder_input = tf.reshape(masked_input, [-1, 10 * 16])

        WC1 = tf.constant(W1, name="WC1")
        BC1 = tf.constant(B1, name="BC1")

        hidden1 = tf.nn.relu(tf.matmul(decoder_input, WC1) + BC1)

        WC2 = tf.constant(W2, name="WC2")
        BC2 = tf.constant(B2, name="BC2")
        hidden2 = tf.nn.relu(tf.matmul(hidden1, WC2) + BC2)
        WC3 = tf.constant(W3, name="WC3")
        BC3 = tf.constant(B3, name="BC3")
        decoder_output = tf.nn.relu(tf.matmul(hidden2, WC3) + BC3, name="output")

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        graph_def = g.as_graph_def()
        tf.train.write_graph(graph_def, EXPORT_DIR, MODEL_NAME, as_text=False)


def predict():
    input_image_batch = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
    batch_size = tf.shape(input_image_batch)[0]

    digitCaps_postRouting, \
    decoder_output, \
    single_digit_prediction, \
    mask_with_labels, \
    correct_labels_placeholder = CapsNetModel.get_model_output_for_tweak(input_image_batch, batch_size)

    all_var = tf.trainable_variables()
    decoder_weights = [var for var in all_var if 'decoder' in var.name]
    for weight_tensor in decoder_weights:
        print(weight_tensor.shape)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # restore
        saver.restore(sess, CHECKPOINT_PATH)

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


predict()
