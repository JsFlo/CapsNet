import os
import numpy as np

import tensorflow as tf
import argparse
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True,
                    help='Path to model (.pb).')
parser.add_argument('--input_array', type=str, required=True,
                    help='Path to array (?, 10, 16, 1) masked digit capsules')
FLAGS = parser.parse_args()


def create_dirs_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def inf():
    # '../example_input_for_decoder_67/example_decoder_67_input.npy'
    a = np.load(FLAGS.input_array)

    with tf.Session() as sess:
        print("load graph")
        with gfile.FastGFile(FLAGS.model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

            print("print graph operations")
            for op in tf.get_default_graph().get_operations():
                print(str(op.name))

            sess.run("init")
            decoder_output = sess.run("output:0", feed_dict={"input:0": np.array([a])})
            print("Decoder output: {}".format(decoder_output.shape))

            plt.figure(figsize=(300, 7))
            plt.subplot(2, 1, 1)
            plt.imshow(decoder_output.reshape([28, 28]), cmap="binary")
            plt.axis("off")
            plt.show()


inf()
