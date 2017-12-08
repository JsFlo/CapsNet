from __future__ import division, print_function, unicode_literals

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf


def printShape(tensor):
    print(tensor.shape)


def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector


def primary_capsule(X):
    printShape(X)  # (?, 28, 28, 1)
    caps1_n_maps = 32
    caps1_n_dims = 8

    conv1 = tf.layers.conv2d(X, filters=256, kernel_size=9, strides=1,
                             padding="valid", activation=tf.nn.relu)
    printShape(conv1)  # (?, 20, 20, 256)

    # stride of 2!
    conv2_n_filters = caps1_n_maps * caps1_n_dims
    conv2 = tf.layers.conv2d(conv1, filters=conv2_n_filters, kernel_size=9, strides=2,
                             padding="valid", activation=tf.nn.relu)
    printShape(conv2)  # (?, 6, 6, 256)

    # what we have: 256 feature maps of 6 x 6 scalar values (total: 9216)
    # what we want: 32 maps of 6x6 vectors (8 dimensions a vector) (total: 9216)

    # BUT since we are going to be FULLY CONNECTING this to the next layer
    # we can just make it one long array [32 * 6 * 6, 8] = [1152, 8] = 1152 x 8 = 9216
    caps1_n_caps = caps1_n_maps * 6 * 6  # 1152 primary capsules
    caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims])
    printShape(caps1_raw)  # (?, 1152, 8)

    # squash to keep the vectors under 1
    return squash(caps1_raw)


def digit_caps(last_layer, batch_size):
    printShape(last_layer)  # (?, 1152, 8)

    # what we have: 1152 primary capsules with 8-dimension vector
    last_layer_n_caps = 1152
    last_layer_n_dims = 8
    # what we want: 10 "digit" capsules with 16-dimension vector
    digit_n_caps = 10
    digit_n_dims = 16

    # The way to go from 8 dimensions to 16 dimensions is to use a **transformation matrix** for each pair (i, j)
    # For each capsule in the first layer *foreach (1, 8) in (1152, 8)* we want to predict the output of every capsule in this layer

    # since we will go from a primary-layer-capsule(1, 8) to a digit-capsule(1, 16)
    # we need the transformation matrix to be (16, 8)
    # ** (16, 8) * (8, 1) = (16, 1)**

    # Transformation matrix for one primary capsule: want [1152, 1] with [16, 8] vectors
    # EFFICIENT: let's do it for all capsules in this layer so we want [1152, 10] with [16, 8] vectors
    # what we want: (?, 1152, 10, 16, 8) "an 1152 by 10 matrix of 16x8 matrix"

    # first lets get (1, 1152, 10, 16, 8)
    init_sigma = 0.01

    W_init = tf.random_normal(
        shape=(1, last_layer_n_caps, digit_n_caps, digit_n_dims, last_layer_n_dims),
        stddev=init_sigma, dtype=tf.float32)
    W = tf.Variable(W_init)

    # copy paste for batch size to get (BATCH_SIZE, 1152, 10, 16, 8)
    W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1])
    printShape(W_tiled)  # (?, 1152, 10, 16, 8)

    # what we have: Transformation_matrix(BATCH_SIZE, 1152, 10, 16, 8)
    # what we need: Second matrix from last layer (BATCH_SIZE, 1152, 10, 8, 1)
    #   last layer matrix: (BATCH_SIZE, 1152, 8)

    # (BATCH_SIZE, 1152, 8) -> (BATCH_SIZE, 1152, 8, 1)
    caps1_output_expanded = tf.expand_dims(last_layer, -1)
    printShape(caps1_output_expanded)

    # (BATCH_SIZE, 1152, 8, 1) -> (BATCH_SIZE, 1152, 1, 8, 1)
    caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2)
    printShape(caps1_output_tile)

    # copy paste for digit_n_caps: (BATCH_SIZE, 1152, 1, 8, 1) -> (BATCH_SIZE, 1152, 10, 8, 1)
    caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, digit_n_caps, 1, 1])
    printShape(caps1_output_tiled)

    digit_caps_predicted = tf.matmul(W_tiled, caps1_output_tiled)
    printShape(digit_caps_predicted)  # (? , 1152, 10, 16, 1)

    return digit_caps_predicted


X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")
batch_size = tf.shape(X)[0]

primaryCapsuleOutput = primary_capsule(X)
digitCapsPredictions = digit_caps(primaryCapsuleOutput, batch_size)



