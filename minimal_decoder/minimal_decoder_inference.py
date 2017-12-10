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
            decoder_output = sess.run("output:0", feed_dict={"input:0": a})
            print("Decoder output: {}".format(decoder_output.shape))

            print("Len decoder output: {}".format(len(decoder_output)))
            n_decoder_output = len(decoder_output)
            idx = 0
            plt.figure(figsize=(300, 7))
            for a_output in decoder_output:
                plt.subplot(2, n_decoder_output, idx + 1)
                plt.imshow(a_output.reshape([28, 28]), cmap="binary")
                plt.axis("off")

                idx += 1
            plt.show()


inf()
