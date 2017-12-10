from __future__ import division, print_function, unicode_literals
import os
import tensorflow as tf

# puts a bad dependency - can only be run from this directory
import sys

sys.path.append('../')

# Makes them look like static method calls (not python style but helps me :)
import caps_net_model.model as CapsNetModel
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True,
                    help='Path to CHECKPOINT_NAME file (without ext) ([CHECKPOINT_NAME].meta, [CHECKPOINT_NAME].index, etc).')
parser.add_argument('--model_output', type=str, required=False, default='./model_output')
parser.add_argument('--model_name', type=str, required=False, default='model_graph.pb')
FLAGS = parser.parse_args()

# CHECKPOINT_PATH = "../checkpoint_with_decoder_tags_67_epoch/my_caps_net"
# EXPORT_DIR = './model_output'
# MODEL_NAME = 'decoder_67_v2.pb'
def create_dirs_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

create_dirs_if_not_exists(FLAGS.model_output)

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
    correct_labels_placeholder = CapsNetModel.get_model_output_for_tweak(input_image_batch, batch_size)

    all_var = tf.trainable_variables()
    decoder_weights = [var for var in all_var if 'decoder' in var.name]
    print("Decoder Weights")
    for weight_tensor in decoder_weights:
        print("Tensors shapes: {}".format(weight_tensor.shape))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # restore
        saver.restore(sess, FLAGS.checkpoint)

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
