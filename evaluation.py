from __future__ import division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils import safe_norm

# Makes them look like static method calls (not python style but helps me :)
import caps_net_model.model as CapsNetModel
import decoder.decoder as Decoder

MNIST = input_data.read_data_sets("/tmp/data/")

CHECKPOINT_PATH = "./checkpoints/my_caps_net"
TRAINING_BATCH_SIZE = 1  # 50

n_iterations_test = MNIST.test.num_examples // TRAINING_BATCH_SIZE


# paper used special margin loss to detect more than 1 digit in an image (overachievers)
def get_margin_loss(predicted_digit, digitCaps_postRouting):
    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5

    predicted_one_hot_digit = tf.one_hot(predicted_digit, depth=10)

    digitCaps_postRouting_safeNorm = safe_norm(digitCaps_postRouting, axis=-2, keep_dims=True)

    present_error_raw = tf.square(tf.maximum(0., m_plus - digitCaps_postRouting_safeNorm))
    present_error = tf.reshape(present_error_raw, shape=(-1, 10))
    absent_error_raw = tf.square(tf.maximum(0., digitCaps_postRouting_safeNorm - m_minus))
    absent_error = tf.reshape(absent_error_raw, shape=(-1, 10))

    loss = tf.add(predicted_one_hot_digit * present_error, lambda_ * (1.0 - predicted_one_hot_digit) * absent_error)
    margin_loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return margin_loss


def get_reconstruction_loss(mask_with_labels, y, y_pred, digitCaps_postRouting, input_image_batch):
    # first take the 10 16-dimension vectors output and pulls out the [predicted digit vector|correct_label digit vector)
    # (ex. prediction: digit 3 so take the 16-dimension vector and pass it to the decoder)

    # make a tensorflow placeholder for choosing based on label or prediction
    # during training pass the correct label digit
    # during inference pass what the model guessed
    reconstruction_targets = tf.cond(mask_with_labels,  # condition
                                     lambda: y,  # if True
                                     lambda: y_pred)  # if False)

    reconstruction_mask = tf.one_hot(reconstruction_targets,
                                     depth=10)

    reconstruction_mask_reshaped = tf.reshape(
        reconstruction_mask, [-1, 1, 10, 1, 1])

    # mask it! (10, 16) * [0, 0, 1, 0, 0, ...]
    masked_out = tf.multiply(digitCaps_postRouting, reconstruction_mask_reshaped)
    # (10, 16) but only (1, 16) has values because of the above


    # Decoder will use the 16 dimension vector to reconstruct the image (28 x 28)
    reconstruction_loss = Decoder.get_reconstruction_loss(masked_out, input_image_batch)
    return reconstruction_loss


def transform_model_output_to_a_single_digit(digitCaps_postRouting):
    # what we have: 10 16-dimensional vectors
    # what we want: which digit are you predicting ?

    # normalize to to get 10 scalars (length of the vectors)
    y_prob = safe_norm(digitCaps_postRouting, axis=-2)
    # (", 1, 10, 1)

    # get index of longest output vector
    y_prob_argmax = tf.argmax(y_prob, axis=2)
    # (?, 1, 1)

    # we have a 1 x 1 matrix , lets just say 1
    y_pred = tf.squeeze(y_prob_argmax, axis=[1, 2])
    return y_pred


def evaluate():
    input_image_batch = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
    batch_size = tf.shape(input_image_batch)[0]

    # prediction
    digitCaps_postRouting = CapsNetModel.get_model_output(input_image_batch, batch_size)
    # (?, 1, 10, 16, 1)

    # single digit prediction
    single_digit_prediction = transform_model_output_to_a_single_digit(digitCaps_postRouting)
    # (?, )

    # labels
    correct_labels_placeholder = tf.placeholder(shape=[None], dtype=tf.int64)

    # loss
    margin_loss = get_margin_loss(correct_labels_placeholder, digitCaps_postRouting)
    mask_with_labels = tf.placeholder_with_default(False, shape=())

    reconstruction_loss = get_reconstruction_loss(mask_with_labels, correct_labels_placeholder, single_digit_prediction,
                                                  digitCaps_postRouting, input_image_batch)

    # keep it small
    reconstruction_alpha = 0.0005
    # favor the margin loss with a small weight for reconstruction loss
    final_loss = tf.add(margin_loss, reconstruction_alpha * reconstruction_loss)

    correct = tf.equal(correct_labels_placeholder, single_digit_prediction)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, CHECKPOINT_PATH)

        loss_tests = []
        acc_tests = []
        for iteration in range(1, n_iterations_test + 1):
            X_batch, y_batch = MNIST.test.next_batch(TRAINING_BATCH_SIZE)

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
        print("hi")
        print("\rFinal test accuracy: {:.4f}%  Loss: {:.6f}".format(
            acc_test * 100, loss_test))
evaluate()