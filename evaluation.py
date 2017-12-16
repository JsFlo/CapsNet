from __future__ import division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import argparse
from tensorflow.examples.tutorials.mnist import input_data

# Makes them look like static method calls (not python style but helps me :)
import caps_net_model.model as CapsNetModel

MNIST = input_data.read_data_sets("/tmp/data/")

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--debug', type=bool, default=False)
FLAGS = parser.parse_args()

if (FLAGS.debug):
    n_iterations_test = 1
    FLAGS.batch_size = 1
else:
    n_iterations_test = MNIST.test.num_examples // FLAGS.batch_size


def evaluate():
    input_image_batch = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
    batch_size = tf.shape(input_image_batch)[0]

    final_loss, accuracy, correct_labels_placeholder = CapsNetModel.get_model_output_for_evaluation(input_image_batch,
                                                                                                    batch_size)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, FLAGS.checkpoint)

        loss_tests = []
        acc_tests = []
        for iteration in range(1, n_iterations_test + 1):
            X_batch, y_batch = MNIST.test.next_batch(FLAGS.batch_size)

            loss_test, acc_test = sess.run(
                [final_loss, accuracy],
                feed_dict={input_image_batch: X_batch.reshape([-1, 28, 28, 1]),
                           correct_labels_placeholder: y_batch})
            loss_tests.append(loss_test)
            acc_tests.append(acc_test)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                iteration, n_iterations_test,
                iteration * 100 / n_iterations_test),
                end=" " * 10)
        loss_test = np.mean(loss_tests)
        acc_test = np.mean(acc_tests)
        print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
            acc_test * 100, loss_test))


evaluate()
