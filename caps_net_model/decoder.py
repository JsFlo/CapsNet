import tensorflow as tf


# interfaces in python ?

def get_reconstruction_loss(capsules, shared_caps_prediction, target_images):
    """

    :param capsules: array of 10 16 dimension vectors
    :param target_images: (? , 28, 28, 1)
    :return: loss
    """
    n_output = 28 * 28

    # flatten - reshape capsules to an array
    capsules_flatten = tf.reshape(capsules, [-1, 10 * 16])
    print("RECONSTRUCTION LOSS - RESHAPED")
    print(capsules_flatten.shape)
    print(shared_caps_prediction.shape)
    shared_caps_flatten = tf.reshape(shared_caps_prediction, [-1, 3])
    final_decoder_input = tf.concat([capsules_flatten, shared_caps_flatten], 1)
    print("RECONSTRUCTION LOSS - END")

    # get prediction array
    decoder_output = _get_tf_layers_impl(final_decoder_input, n_output)
    # decoder_output = _get_tf_nn_impl(decoder_input, n_output)

    # flatten - reshape target images from (28 x 28) to (784)
    X_flat = tf.reshape(target_images, [-1, n_output])
    squared_difference = tf.square(X_flat - decoder_output)
    reconstruction_loss = tf.reduce_sum(squared_difference)

    return reconstruction_loss, decoder_output


def _get_tf_layers_impl(decoder_input, target,
                        n_hidden1=512, n_hidden2=1024):
    """
    3 fc layers
    512 -> relu ->
    1024 -> relu ->
    target -> sigmoid ->

    :param decoder_input:
    :return:
    """
    with tf.variable_scope('decoder') as scope:
        n_output = target
        hidden1 = tf.layers.dense(decoder_input, n_hidden1,
                                  activation=tf.nn.relu)
        hidden2 = tf.layers.dense(hidden1, n_hidden2,
                                  activation=tf.nn.relu)
        decoder_output = tf.layers.dense(hidden2, n_output,
                                         activation=tf.nn.sigmoid)
        return decoder_output


def _get_tf_nn_impl(decoder_input, target,
                    n_hidden1=512, n_hidden2=1024):
    with tf.variable_scope('decoder') as scope:
        # 160 was flattened above (10 x 16)
        fc1 = getFullyConnectedLayer_relu(decoder_input, 163, n_hidden1)
        fc2 = getFullyConnectedLayer_relu(fc1, n_hidden1, n_hidden2)
        fc3 = getFullyConnectedLayer_sigmoid(fc2, n_hidden2, target)
        return fc3


# fully connected with relu
def getFullyConnectedLayer_relu(lastLayer, input, output):
    W_fc1 = weight_variable([input, output])
    b_fc1 = bias_variable([output])

    return tf.nn.relu(tf.matmul(lastLayer, W_fc1) + b_fc1)


# fully connected with sigmoid
def getFullyConnectedLayer_sigmoid(lastLayer, input, output):
    W_fc1 = weight_variable([input, output])
    b_fc1 = bias_variable([output])

    return tf.nn.sigmoid(tf.matmul(lastLayer, W_fc1) + b_fc1)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
