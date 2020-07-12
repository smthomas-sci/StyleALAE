"""

Losses for training an ALAE

Author: Simon Thomas
Date: Jul-06-2020

"""

import tensorflow.keras.backend as K

from tensorflow.keras.activations import softplus as f


def l2(x_true, x_pred):
    """
    L2 for the reciprocal loss in latent space

    :param - x_true - groundtruth
    :param - x_pred - prediction
    """
    return K.mean((x_true - x_pred) ** 2)


def discriminator_logistic_non_saturating(d_real, d_fake):
    """
    Discriminator loss, where f = softplus.

    :param - d_real - discriminator real output
    :param - d_fake - discriminator real output
    """
    loss = f(-d_real) + f(d_fake)

    return K.mean(loss)


def generator_logistic_non_saturating(g_result):
    """
    generator loss, where f = softplus
    """
    loss = f(-g_result)
    return K.mean(loss)