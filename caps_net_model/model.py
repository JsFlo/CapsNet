from __future__ import division, print_function, unicode_literals

import tensorflow as tf

import caps_net_model.digit_capsules as DigitCapsules
import caps_net_model.primary_capsules as PrimaryCapsules
import caps_net_model.decoder as decoder
# Makes them look like static method calls (not python style but helps me :)
from utils import safe_norm


def _get_model_output(input_image_batch, batch_size):
    primaryCapsules = PrimaryCapsules.get_primary_capsules(input_image_batch)

    # prediction
    digitCaps_postRouting, shared_caps_prediction = DigitCapsules.get_digit_caps_output(primaryCapsules, batch_size)
    # (?, 1, 10, 16, 1)

    # single digit prediction
    single_digit_prediction = _transform_model_output_to_a_single_digit(digitCaps_postRouting)
    # (?, )

    # labels
    correct_labels_placeholder = tf.placeholder(shape=[None], dtype=tf.int64)

    # loss
    margin_loss = _get_margin_loss(correct_labels_placeholder, digitCaps_postRouting)
    mask_with_labels = tf.placeholder_with_default(False, shape=())

    reconstruction_loss, decoder_output, masked_out = _get_reconstruction_loss(mask_with_labels,
                                                                               correct_labels_placeholder,
                                                                               single_digit_prediction,
                                                                               digitCaps_postRouting, input_image_batch)

    # keep it small
    reconstruction_alpha = 0.0005
    # favor the margin loss with a small weight for reconstruction loss
    final_loss = tf.add(margin_loss, reconstruction_alpha * reconstruction_loss)

    correct = tf.equal(correct_labels_placeholder, single_digit_prediction)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(final_loss)

    return digitCaps_postRouting, final_loss, correct, \
           accuracy, optimizer, training_op, mask_with_labels, \
           decoder_output, single_digit_prediction, correct_labels_placeholder, masked_out


def get_model_output_for_training(input_image_batch, batch_size):
    digitCaps_postRouting, final_loss, correct, \
    accuracy, optimizer, training_op, mask_with_labels, \
    decoder_output, single_digit_prediction, correct_labels_placeholder, masked_out = _get_model_output(input_image_batch,
                                                                                            batch_size)

    return digitCaps_postRouting, final_loss, correct, \
           accuracy, optimizer, training_op, mask_with_labels, \
           correct_labels_placeholder


def get_model_output_for_evaluation(input_image_batch, batch_size):
    digitCaps_postRouting, final_loss, correct, \
    accuracy, optimizer, training_op, mask_with_labels, \
    decoder_output, single_digit_prediction, correct_labels_placeholder, masked_out = _get_model_output(input_image_batch,
                                                                                            batch_size)

    return final_loss, accuracy, correct_labels_placeholder


def get_model_output_for_predictions(input_image_batch, batch_size):
    digitCaps_postRouting, final_loss, correct, \
    accuracy, optimizer, training_op, mask_with_labels, \
    decoder_output, single_digit_prediction, correct_labels_placeholder, masked_out = _get_model_output(input_image_batch,
                                                                                            batch_size)
    return digitCaps_postRouting, \
           decoder_output, \
           single_digit_prediction, \
           correct_labels_placeholder, masked_out


def get_model_output_for_tweak(input_image_batch, batch_size):
    digitCaps_postRouting, final_loss, correct, \
    accuracy, optimizer, training_op, mask_with_labels, \
    decoder_output, single_digit_prediction, correct_labels_placeholder, masked_out = _get_model_output(input_image_batch,
                                                                                            batch_size)
    return digitCaps_postRouting, \
           decoder_output, \
           single_digit_prediction, \
           mask_with_labels, \
           correct_labels_placeholder, masked_out


# paper used special margin loss to detect more than 1 digit in an image (overachievers)
def _get_margin_loss(predicted_digit, digitCaps_postRouting):
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


def _get_reconstruction_loss(mask_with_labels, y, y_pred, digitCaps_postRouting, input_image_batch):
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
    # print("shape!!!!!!: {}".format(masked_out))
    # masked out
    # (10, 16) but only (1, 16) has values because of the above


    # Decoder will use the 16 dimension vector to reconstruct the image (28 x 28)
    reconstruction_loss, decoder_output = decoder.get_reconstruction_loss(masked_out, input_image_batch)
    return reconstruction_loss, decoder_output, masked_out


def _transform_model_output_to_a_single_digit(digitCaps_postRouting):
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
