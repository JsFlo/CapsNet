import os
import numpy as np

import tensorflow as tf
import utils
import argparse
from tensorflow.python.platform import gfile
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True,
                    help='Path to model checkpoint.')
# parser.add_argument('--output_path', type=str, default='./inf_out/images/',
#                     help='Path to model checkpoint.')
# parser.add_argument('--output_image_columns', type=int, default=4)
# parser.add_argument('--output_num_images', type=int, default=16)
FLAGS = parser.parse_args()

GEN_INPUT_DIM = 100


def create_dirs_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_train_noise():
    return np.random.uniform(-1.0, 1.0, size=[FLAGS.output_num_images, GEN_INPUT_DIM]).astype(np.float32)


def get_counter():
    count = len([name for name in os.listdir(FLAGS.output_path)])
    print("count: {}".format(count))
    return count


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# fully connected with relu
def getFullyConnectedLayer_relu(lastLayer, input, output):
    W_fc1 = weight_variable([input, output])
    b_fc1 = bias_variable([output])

    return tf.nn.relu(tf.matmul(lastLayer, W_fc1) + b_fc1)


# fully connected with sigmoid
def getFullyConnectedLayer_sigmoid(lastLayer, input, output):
    W_fc1 = weight_variable([input, output])
    b_fc1 = bias_variable([output])

    return tf.nn.sigmoid(tf.matmul(lastLayer, W_fc1) + b_fc1, name="output")


def _get_tf_nn_impl(decoder_input, target,
                    n_hidden1=512, n_hidden2=1024):
    # 160 was flattened above (10 x 16)
    fc1 = getFullyConnectedLayer_relu(decoder_input, 160, n_hidden1)
    fc2 = getFullyConnectedLayer_relu(fc1, n_hidden1, n_hidden2)
    fc3 = getFullyConnectedLayer_sigmoid(fc2, n_hidden2, target)
    return fc3


def inf():
    # masked_input = tf.placeholder(dtype="float32", shape=[None, 1, 10, 16, 1], name="input")
    # decoder_input = tf.reshape(masked_input, [-1, 10 * 16])
    # decoder_output = _get_tf_nn_impl(decoder_input, 28 * 28)

    # sess init
    # sess = tf.Session()
    # saver = tf.train.Saver()
    a = np.load('example_input_for_decoder_67/example_decoder_67_input.npy')
    # saver.restore(sess, FLAGS.model_path)

    # with tf.Session(graph=tf.Graph()) as sess:
    #     tf.saved_model.loader.load(
    #         sess,
    #         [tf.saved_model.tag_constants.SERVING],
    #         FLAGS.model_path)
    #init_op = tf.initialize_all_variables()

    with tf.Session() as sess:
        print("load graph")
        with gfile.FastGFile(FLAGS.model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            for op in tf.get_default_graph().get_operations():
                print(str(op.name))
            #sess.run(init_op)

            print(np.array([a]))
            sess.run("init")
            #imgtest = sess.run("output")
            imgtest = sess.run("output:0", feed_dict={"input:0": np.array([a])})
            print(imgtest)
            #
            plt.figure(figsize=(300, 7))
            plt.subplot(2, 1, 1)
            plt.imshow(imgtest.reshape([28, 28]), cmap="binary")
            plt.axis("off")
            plt.show()


inf()
