import tensorflow as tf
from utils import squash


def get_primary_capsules(X):
    # print("Primary Capsules")
    # printShape(X)  # (?, 28, 28, 1)
    caps1_n_maps = 32
    caps1_n_dims = 8

    conv1 = tf.layers.conv2d(X, filters=256, kernel_size=9, strides=1,
                             padding="valid", activation=tf.nn.relu)
    # printShape(conv1)  # (?, 20, 20, 256)

    # stride of 2!
    conv2_n_filters = caps1_n_maps * caps1_n_dims
    conv2 = tf.layers.conv2d(conv1, filters=conv2_n_filters, kernel_size=9, strides=2,
                             padding="valid", activation=tf.nn.relu)
    # printShape(conv2)  # (?, 6, 6, 256)

    # what we have: 256 feature maps of 6 x 6 scalar values (total: 9216)
    # what we want: 32 maps of 6x6 vectors (8 dimensions a vector) (total: 9216)

    # BUT since we are going to be FULLY CONNECTING this to the next layer
    # we can just make it one long array [32 * 6 * 6, 8] = [1152, 8] = 1152 x 8 = 9216
    caps1_n_caps = caps1_n_maps * 6 * 6  # 1152 primary capsules
    caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims])
    # printShape(caps1_raw)  # (?, 1152, 8)

    # squash to keep the vectors under 1
    return squash(caps1_raw)
