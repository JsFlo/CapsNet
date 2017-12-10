import tensorflow as tf
import numpy as np

from utils import squash


def get_digit_caps_output(last_layer, batch_size):
    # "primary capsules digit caps prediction" (1152, 10, 16)
    prediction_of_digit_caps = _get_primary_cap_prediction_of_digit_caps(last_layer, batch_size)

    # routing by agreement (may involve multiple rounds)
    digit_caps_predictions = _routing_by_agreement(prediction_of_digit_caps, batch_size)
    return digit_caps_predictions


def _get_primary_cap_prediction_of_digit_caps(last_layer, batch_size):
    """
    Takes the last layer capsules  (8, 1) and multiplies a transformation matrix for
    each pair (i, j) of shape (16, 8)

    What we have: 1152 primary capsules with 8-dimension vector
    What we want: 10 "digit" capsules with 16-dimension vector


    :param last_layer: (?, 1152, 8)
    :param batch_size: batch size
    :return:
    """

    last_layer_n_caps = 1152
    last_layer_n_dims = 8

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
    # (?, 1152, 10, 16, 8)

    # what we have: Transformation_matrix(BATCH_SIZE, 1152, 10, 16, 8)
    # what we need: Second matrix from last layer (BATCH_SIZE, 1152, 10, 8, 1)
    #   last layer matrix: (BATCH_SIZE, 1152, 8)

    # (BATCH_SIZE, 1152, 8) -> (BATCH_SIZE, 1152, 8, 1)
    caps1_output_expanded = tf.expand_dims(last_layer, -1)

    # (BATCH_SIZE, 1152, 8, 1) -> (BATCH_SIZE, 1152, 1, 8, 1)
    caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2)

    # copy paste for digit_n_caps: (BATCH_SIZE, 1152, 1, 8, 1) -> (BATCH_SIZE, 1152, 10, 8, 1)
    caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, digit_n_caps, 1, 1])

    digit_caps_predicted = tf.matmul(W_tiled, caps1_output_tiled)
    # (? , 1152, 10, 16, 1)

    return digit_caps_predicted


def _routing_by_agreement(digit_caps, batch_size):
    # digitCaps (?, 1152, 10, 16, 1)

    # weight for every pair
    raw_weights = tf.zeros([batch_size, digit_caps.shape[1].value, digit_caps.shape[2].value, 1, 1],
                           dtype=np.float32)
    # (?, 1152, 10, 1, 1)

    # round 1
    round1_output = _routing_round(raw_weights, digit_caps)
    round1_agreement = _get_round_agreement(round1_output, digit_caps, 1152)

    # now update those weights from the beginning
    # by just adding the agreement values
    raw_weights_round_2 = tf.add(raw_weights, round1_agreement)

    round2_output = _routing_round(raw_weights_round_2, digit_caps)
    return round2_output


def _routing_round(previous_weights, digit_caps_prediction):
    # print(": routing weights = softmax on previous weights")
    routing_weights = tf.nn.softmax(previous_weights, dim=2)
    # (?, 1152, 10, 1, 1)

    # print(": weighted predictions = routing weights x digit caps prediction")
    weighted_predictions = tf.multiply(routing_weights, digit_caps_prediction)
    # (?, 1152, 10, 16, 1)

    # Q: When getting weighted predictions why is there no bias ?

    # print(": reduce sum of all of them (collapse `rows`)")
    weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True)
    # (?, 1 , 10, 16, 1)

    # print(": squash to keep below 1")
    round_output = squash(weighted_sum, axis=-2)
    # (?, 1 , 10, 16, 1)
    return round_output


def _get_round_agreement(round_output, digit_capsule_output, primary_n_caps):
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
    :return: (?, 1152, 10,1, 1)
    """

    # the copy&paste
    round_output_tiled = tf.tile(
        round_output, [1, primary_n_caps, 1, 1, 1])

    # that scalar product we talked about above
    agreement = tf.matmul(digit_capsule_output, round_output_tiled, transpose_a=True)
    # (?, 1152, 10, 1, 1)

    return agreement
