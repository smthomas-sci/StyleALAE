"""

Losses for training an ALAE

Author: Simon Thomas
Date: Jul-06-2020

"""

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.activations import softplus as f


@tf.function
def l2(x_true, x_pred):
    """
    L2 for the reciprocal loss in latent space

    :param - x_true - groundtruth
    :param - x_pred - prediction
    """
    return tf.nn.compute_average_loss((x_true - x_pred) ** 2)


@tf.function
def discriminator_logistic_non_saturating(d_real, d_fake):
    """
    Discriminator loss, where f = softplus.

    :param - d_real - discriminator real output
    :param - d_fake - discriminator real output
    """
    loss = f(-d_real) + f(d_fake)

    return tf.nn.compute_average_loss(loss)


@tf.function
def generator_logistic_non_saturating(g_result):
    """
    generator loss, where f = softplus
    """
    loss = f(-g_result)
    return tf.nn.compute_average_loss(loss)