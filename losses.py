"""

Losses for training an ALAE

Author: Simon Thomas
Date: Jul-06-2020

"""

import tensorflow as tf
from tensorflow.keras.activations import softplus as f


class L2(tf.keras.losses.Loss):
    def __init__(self):
        super(L2).__init__(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred, bs=None):
        return tf.nn.compute_average_loss((y_true - y_pred) ** 2, global_batch_size=bs)


class DNS(tf.keras.losses.Loss):
    def __init__(self):
        super(DNS).__init__(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, d_real, d_fake, bs=None):
        loss = f(-d_real) + f(d_fake)

        return tf.nn.compute_average_loss(loss, global_batch_size=bs)


class GNS(tf.keras.losses.Loss):
    def __init__(self):
        super(GNS).__init__(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, g_result, bs=None):
        loss = f(-g_result)
        return tf.nn.compute_average_loss(loss, global_batch_size=bs)

# Create losses
l2 = L2()
discriminator_logistic_non_saturating = DNS()
generator_logistic_non_saturating = GNS()


# def l2(x_true, x_pred, bs=None):
#     """
#     L2 for the reciprocal loss in latent space
#
#     :param - x_true - groundtruth
#     :param - x_pred - prediction
#     """
#     return tf.nn.compute_average_loss((x_true - x_pred) ** 2, global_batch_size=bs)


# def discriminator_logistic_non_saturating(d_real, d_fake, bs=None):
#     """
#     Discriminator loss, where f = softplus.
#
#     :param - d_real - discriminator real output
#     :param - d_fake - discriminator real output
#     """
#     loss = f(-d_real) + f(d_fake)
#
#     return tf.nn.compute_average_loss(loss, global_batch_size=bs)


# def generator_logistic_non_saturating(g_result, bs=None):
#     """
#     generator loss, where f = softplus
#     """
#     loss = f(-g_result)
#     return tf.nn.compute_average_loss(loss, global_batch_size=bs)
