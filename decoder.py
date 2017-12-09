import numpy as np
import tensorflow as tf


# interfaces in python ?

def get_reconstruction_loss(capsules, target_images):
    """

    :param capsules: array of 10 16 dimension vectors
    :param target_images: (? , 28, 28, 1)
    :return: loss
    """
    n_output = 28 * 28

    # flatten - reshape capsules to an array
    decoder_input = tf.reshape(capsules, [-1, 10 * 16])

    # get prediction array
    decoder_output = get_tf_layers_impl(decoder_input, n_output)

    # flatten - reshape target images from (28 x 28) to (784)
    X_flat = tf.reshape(target_images, [-1, n_output])
    squared_difference = tf.square(X_flat - decoder_output)
    reconstruction_loss = tf.reduce_sum(squared_difference)

    return reconstruction_loss


def get_tf_layers_impl(decoder_input, target,
                       n_hidden1=512, n_hidden2=1024):
    """
    3 fc layers
    512 -> relu ->
    1024 -> relu ->
    target -> sigmoid ->

    :param decoder_input:
    :return:
    """
    n_output = target

    hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                              activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2,
                              activation=tf.nn.relu)
    decoder_output = tf.layers.dense(hidden2, n_output,
                                     activation=tf.nn.sigmoid)
    return decoder_output
