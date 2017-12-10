from __future__ import division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils import printShape
from utils import safe_norm

# Makes them look like static method calls (not python style but helps me :)
import caps_net_model.model as CapsNetModel
import decoder.decoder as Decoder

MNIST = input_data.read_data_sets("/tmp/data/")

CHECKPOINT_PATH = "./checkpoints/my_caps_net"
RESTORE_CHECKPOINT = True

TRAINING_BATCH_SIZE = 1  # 50
N_EPOCHS = 1  # 10
N_ITERATIONS_PER_EPOCH = 1  # mnist.train.num_examples // TRAINING_BATCH_SIZE
N_ITERATIONS_VALIDATION = 1  # mnist.validation.num_examples // TRAINING_BATCH_SIZE

BEST_LOSS_VAL = np.infty


def train():
    input_image_batch = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
    batch_size = tf.shape(input_image_batch)[0]

    digitCaps_postRouting = CapsNetModel.get_model_output(input_image_batch, batch_size)

    print("\nDigit caps post routing")
    printShape(digitCaps_postRouting)  # (?, 1, 10, 16, 1)
    print(": ")

    # what we have: 10 16-dimensional vectors
    # what we want: which digit are you predicting ?

    # normalize to to get 10 scalars (length of the vectors)
    y_prob = safe_norm(digitCaps_postRouting, axis=-2)
    printShape(y_prob)  # (", 1, 10, 1)

    # get index of longest output vector
    y_prob_argmax = tf.argmax(y_prob, axis=2)
    printShape(y_prob_argmax)  # (?, 1, 1)

    # we have a 1 x 1 matrix , lets just say 1
    y_pred = tf.squeeze(y_prob_argmax, axis=[1, 2])
    printShape(y_pred)  # (?, )

    print("\nLoss")

    # labels
    y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")

    # paper used special margin loss to detect more than 1 digit in an image (overachievers)
    m_plus = 0.9
    m_minus = 0.1
    lambda_ = 0.5

    T = tf.one_hot(y, depth=10, name="T")

    # again
    digitCaps_postRouting_safeNorm = safe_norm(digitCaps_postRouting, axis=-2, keep_dims=True)

    present_error_raw = tf.square(tf.maximum(0., m_plus - digitCaps_postRouting_safeNorm))
    present_error = tf.reshape(present_error_raw, shape=(-1, 10))
    absent_error_raw = tf.square(tf.maximum(0., digitCaps_postRouting_safeNorm - m_minus))
    absent_error = tf.reshape(absent_error_raw, shape=(-1, 10))

    loss = tf.add(T * present_error, lambda_ * (1.0 - T) * absent_error)
    margin_loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))

    # reconstruction - there will be a loss here too used to compute the final loss
    # the final loss takes into account the margin loss and the reconstruction loss

    # first take the 10 16-dimension vectors output and pulls out the [predicted digit vector|correct_label digit vector)
    # (ex. prediction: digit 3 so take the 16-dimension vector and pass it to the decoder)

    # make a tensorflow placeholder for choosing based on label or prediction
    # during training pass the correct label digit
    # during inference pass what the model guessed
    mask_with_labels = tf.placeholder_with_default(False, shape=())

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
    n_output = 28 * 28
    reconstruction_loss = Decoder.get_reconstruction_loss(masked_out, input_image_batch)

    # keep it small
    reconstruction_alpha = 0.0005
    # favor the margin loss with a small weight for reconstruction loss
    final_loss = tf.add(margin_loss, reconstruction_alpha * reconstruction_loss)

    correct = tf.equal(y, y_pred)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(final_loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    best_loss_val = BEST_LOSS_VAL
    with tf.Session() as sess:
        if RESTORE_CHECKPOINT and tf.train.checkpoint_exists(CHECKPOINT_PATH):
            saver.restore(sess, CHECKPOINT_PATH)
        else:
            init.run()

        for epoch in range(N_EPOCHS):
            for iteration in range(1, N_ITERATIONS_PER_EPOCH + 1):
                X_batch, y_batch = MNIST.train.next_batch(TRAINING_BATCH_SIZE)
                # Run the training operation and measure the loss:
                _, loss_train = sess.run([training_op, final_loss],
                                         feed_dict={input_image_batch: X_batch.reshape([-1, 28, 28, 1]),
                                                    y: y_batch,
                                                    mask_with_labels: True})  # use labels during training for the decoder

                print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                    iteration, N_ITERATIONS_PER_EPOCH,
                    iteration * 100 / N_ITERATIONS_PER_EPOCH,
                    loss_train),
                    end="")

            # At the end of each epoch,
            # measure the validation loss and accuracy:
            loss_vals = []
            acc_vals = []
            for iteration in range(1, N_ITERATIONS_VALIDATION + 1):
                X_batch, y_batch = MNIST.validation.next_batch(TRAINING_BATCH_SIZE)
                loss_val, acc_val = sess.run(
                    [final_loss, accuracy],
                    feed_dict={input_image_batch: X_batch.reshape([-1, 28, 28, 1]),
                               y: y_batch})
                loss_vals.append(loss_val)
                acc_vals.append(acc_val)
                print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                    iteration, N_ITERATIONS_VALIDATION,
                    iteration * 100 / N_ITERATIONS_VALIDATION),
                    end=" " * 10)
            loss_val = np.mean(loss_vals)
            acc_val = np.mean(acc_vals)
            print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}".format(
                epoch + 1, acc_val * 100, loss_val))

            # save if improved
            if loss_val < best_loss_val:
                print("(improved)")
                save_path = saver.save(sess, CHECKPOINT_PATH)
                best_loss_val = loss_val
train()
