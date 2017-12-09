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


def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                     keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)


def primary_capsule(X):
    print("Primary Capsules")
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
    print("Digit Capsules")
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


def routing_by_agreement(digit_caps):
    print("Routing by Agreement")
    printShape(digit_caps)  # (?, 1152, 10, 16, 1)

    # weight for every pair
    raw_weights = tf.zeros([batch_size, digit_caps.shape[1].value, digit_caps.shape[2].value, 1, 1],
                           dtype=np.float32)
    print("raw weights shape: {}".format(raw_weights.shape))  # (?, 1152, 10, 1, 1)

    # round 1
    round1_output = routing_round(raw_weights, digit_caps)
    round1_agreement = get_round_agreement(round1_output, digit_caps, 1152)

    # now update those weights from the beginning
    # by just adding the agreement values
    raw_weights_round_2 = tf.add(raw_weights, round1_agreement)

    round2_output = routing_round(raw_weights_round_2, digit_caps)
    return round2_output


def routing_round(previous_weights, digit_caps_prediction):
    print("\nRouting Round")
    print(": Previous weights")
    printShape(previous_weights)
    print(": Digit caps prediction")
    printShape(digit_caps_prediction)
    print(": ")

    print(": routing weights = softmax on previous weights")
    routing_weights = tf.nn.softmax(previous_weights, dim=2)

    print(": weighted predictions = routing weights x digit caps prediction")
    weighted_predictions = tf.multiply(routing_weights, digit_caps_prediction)
    printShape(weighted_predictions)

    # Q: When getting weighted predictions why is there no bias ?

    print(": reduce sum of all of them (collapse `rows`)")
    weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True)
    printShape(weighted_sum)

    print(": squash to keep below 1")
    round_output = squash(weighted_sum, axis=-2)
    return round_output


def get_round_agreement(round_output, digit_capsule_output, primary_n_caps):
    """
    Measure how close the digit capsule output is compared to a round guess.

    How to measure for **1** capsule:
        A = From digit-capsule_output (?, 1152, 10, 16, 1) take one(1, 10, 16, 1) take (10, 16)
        B = From round output (1, 10, 16, 1) take (10, 16)

        SCALAR = Perform a scalar product A(transpose) dot B

        The SCALAR value is the "agreement" value

        WAIT! Lets do all these scalar products in one run:
        digit-capsule_output (?, 1152, 10, 16, 1)
        round output (1, 10, 16, 1)

        COPY & PASTE the round output to match the digit_capsule_output
        so we want round output to be (1152, 10, 16, 1)

    :param round_output: (1, 10, 16, 1)
    :param digit_capsule_output: (?, 1152, 10, 16, 1)
    :param primary_n_caps: 1152
    :return:
    """
    print("\nRound Agreement")
    print(": Round Output")
    printShape(round_output)
    print(": Digit capsule output")
    printShape(digit_capsule_output)
    print(": ")

    print(": Tile round output")
    # the copy&paste
    round_output_tiled = tf.tile(
        round_output, [1, primary_n_caps, 1, 1, 1])
    printShape(round_output_tiled)

    print(": Scalar Product")
    # that scalar product we talked about above
    agreement = tf.matmul(digit_capsule_output, round_output_tiled, transpose_a=True)
    printShape(agreement)

    return agreement


X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")
batch_size = tf.shape(X)[0]

primaryCapsuleOutput = primary_capsule(X)
# this is more like "primary capsules digit caps prediction" (1152, 10, 16)
digitCapsPredictions = digit_caps(primaryCapsuleOutput, batch_size)
# and this is actually the digit caps output of (10, 16)
digitCaps_postRouting = routing_by_agreement(digitCapsPredictions)

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


