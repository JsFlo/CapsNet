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
RESTORE_CHECKPOINT = True

TRAINING_BATCH_SIZE = 1  # 50
N_EPOCHS = 1  # 10
N_ITERATIONS_PER_EPOCH = 1  # mnist.train.num_examples // TRAINING_BATCH_SIZE
N_ITERATIONS_VALIDATION = 1  # mnist.validation.num_examples // TRAINING_BATCH_SIZE

BEST_LOSS_VAL = np.infty


def train():
    input_image_batch = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32)
    batch_size = tf.shape(input_image_batch)[0]

    # prediction
    digitCaps_postRouting, \
    final_loss, correct, accuracy, \
    optimizer, training_op, mask_with_labels, \
    correct_labels_placeholder = CapsNetModel.get_model_output_for_training(input_image_batch, batch_size)

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
                # train and get loss to log
                _, loss_train = sess.run([training_op, final_loss],
                                         feed_dict={input_image_batch: X_batch.reshape([-1, 28, 28, 1]),
                                                    correct_labels_placeholder: y_batch,
                                                    mask_with_labels: True})  # use labels during training for the decoder

                print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(
                    iteration, N_ITERATIONS_PER_EPOCH,
                    iteration * 100 / N_ITERATIONS_PER_EPOCH,
                    loss_train),
                    end="")

            # check against validation set and log it
            loss_vals = []
            acc_vals = []
            for iteration in range(1, N_ITERATIONS_VALIDATION + 1):
                X_batch, y_batch = MNIST.validation.next_batch(TRAINING_BATCH_SIZE)
                loss_val, acc_val = sess.run(
                    [final_loss, accuracy],
                    feed_dict={input_image_batch: X_batch.reshape([-1, 28, 28, 1]),
                               correct_labels_placeholder: y_batch})
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
