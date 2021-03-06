"""
Metrics for determining the quality of generated and reconstructed images.

Author: Simon Thomas
Date: 28-07-2020
Updated: 10-11-2020

"""
import numpy as np
import os
import tensorflow as tf

from StyleALAE.data import *

# FID
# from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_iv3
import tensorflow_hub as tfhub

# PPL
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg

from tensorflow.keras.models import Model

from skimage.transform import resize
from skimage import io
from scipy.linalg import sqrtm


def _generator(path, k, model="iv3", batch_size=12):
    """
    Image generator
    :param path: path to files
    :param k: how many images to include
    :para model: which model to use: vgg or iv3?
    :yield: image
    """
    preprocess_input = preprocess_input_vgg if model=="vgg" else preprocess_input_iv3
    size = (224, 224) if model == "vgg" else None
    files = [os.path.join(path, file) for file in os.listdir(path)][0:k]
    n_batches = k // batch_size
    i = 0
    while i < n_batches:
        batch = [ io.imread(fn).astype("float32") for fn in files[i*n_batches:i*n_batches + batch_size]]
        if model == "vgg": # auto-resize for inception
            batch = tf.image.resize(batch, size)
        batch = preprocess_input(batch)
        yield np.expand_dims(batch, 0)


class PPL(object):
    """
    Calculates the Perceptual Path Length (PPL) for determining the perceptual changes
    in images resulting from interpolation in the latent space.

    In z space there two points generated (z_1, z_2), between which there are infinite real
    intermediaries. In practice, the linear space is discretised with e = 10E-04.

            z_1                                 z_2
             • - | - | - | - | - | - | - | - | - •
               z_1+e                       z_2-e

    Each point can be fed to the generator (G) to create an image x.
    Using VGG perceptual space, we can see how an image changes by traversing the line
    between z_1 and z_2 in increments of e. Distance is defined as at
    https://arxiv.org/pdf/1801.03924.pdf and intuitively is

        D = SUM([ 1/(H_lxW_l) VGG19(G(z))_l, VGG19(G(z+e))_l for l in LAYERS])

    Notes:
        - perceptual weights taken from The Unreasonable Effectiveness of Deep Features as a Perceptual Metric,
        using the pytorch version https://github.com/richzhang/PerceptualSimilarity


    """
    def __init__(self, generator, k=10000):
        """

        :param img_path:
        :param generator:
        :param k:
        """
        self.generator = generator
        self.k = k
        self.dist = self._create_perceptual_model()

    def _normalize(self, v):
        return v / tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1, keepdims=True))

    def _create_perceptual_model(self):
        """
        According to https://arxiv.org/pdf/1707.09405.pdf the following
        5 layers are used in the VGG19 model:
            [‘conv1 2’, ‘conv2 2’, conv3 2’, ‘conv4 2’, ‘conv5 2’]
        :return:
        """
        vgg = VGG19(include_top=False,
                           weights="imagenet",
                           input_shape=(224, 224, 3),
                           pooling="avg",
                    )
        # Get input
        feat_in = vgg.get_input_at(0)

        # Get features
        features = [vgg.get_layer(f"block{b+1}_conv2").output for b in range(5)]

        # Normalise
        y_hats = []
        for x in features:
            y_hats.append(self._normalize(x))

        # Scale
        scaled = []
        for b, (x, f) in enumerate(zip(y_hats, [64, 128, 256, 512, 512])):
            x *= np.load(f"./weights/perceptual_similarity_1x1_weights/block_{b + 1}.npy")
            scaled.append(x)

        # L2 + Avg
        avgs = []
        for feat in scaled:
            H, W = feat[0].shape[0:2]
            l2 = tf.reduce_sum(tf.square(feat[0] - feat[1]), keepdims=True)
            avg = l2*(1/(H*W))
            avgs.append(avg)
        distance = tf.keras.layers.Add(name="Sum")(avgs)

        model = Model(inputs=[feat_in], outputs=distance)

        return model

    def _lerp(self, w_1, w_2, α):
        """
        Linear interpolation
        :param w_1: w point 1
        :param w_2: w point 2
        :param α: alpha weighting for vectors - np.arange(0, 1, e)
        :return: interpolated z
        """
        w_1 = self._normalize(w_1)
        w_2 = self._normalize(w_2)
        return (1-α)*w_1 + α*w_2

    def _slerp(self, z_1, z_2, α):
        """
        Spherical linear interpolation - use with gaussian prior
        refs:
         - https://github.com/NVlabs/stylegan/blob/master/metrics/perceptual_path_length.py
         - https://en.wikipedia.org/wiki/Slerp

        :param z_1: z point 1
        :param z_2: z points 2
        :param α: alpha weighting for vectors - np.arange(0, 1, e)
        :return: interpolated z
        """
        z_1 = self._normalize(z_1)
        z_2 = self._normalize(z_2)
        dot = tf.reduce_sum(z_1 * z_2, axis=-1, keepdims=True)
        theta = α * tf.math.acos(dot)
        c = self._normalize(z_2 - dot * z_1)
        interp = z_1 * tf.math.cos(theta) + c * tf.math.sin(theta)
        return self._normalize(interp)

    def get_latent_interpolations(self, space, z_dim=2, e=10e-4):
        """

        :param space: whether to interpolate in "z" or "w" space
        :param e: the step size
        :return: ppl_score
        """

        if space == "z":
            interp = self._slerp
        if space == "w":
            interp = self._lerp

        # Generate random z pair
        z1 = tf.Variable(tf.random.normal((1, z_dim)), dtype=tf.float32)
        z2 = tf.Variable(tf.random.normal((1, z_dim)), dtype=tf.float32)

        interps = []
        for a in np.arange(0, 1+e, e):
            interps.append(interp(z1, z2, a))

        return interps

    def compute_distance_sum(self, zs):
        distances = []
        for i in range(len(zs)-1):
            x1 = self.generator(zs[i])
            x2 = self.generator(zs[i+1])
            # Scale dynamic range for VGG [0,255]
            xs = tf.concat([x1, x2], axis=0) * 255.
            xs = preprocess_input_vgg(xs)
            d = self.dist(xs)
            distances.append(d)
        distances = np.stack([d.numpy().squeeze() for d in distances])
        return np.sum(distances)

if __name__ == "__main__":

    print("hello world!")

    # -------- FAKE GENERATOR ------------------ #
    gen_in = tf.keras.layers.Input(shape=(49,))
    x = tf.keras.layers.Reshape((7, 7, 1))(gen_in)
    x = tf.keras.layers.Conv2D(4, kernel_size=(3, 3), padding="same")(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(3, 1)(x)
    gen_out = x
    gen = tf.keras.models.Model(inputs=[gen_in], outputs=[gen_out])
    # ---------------------------------------------------- #

    N = 100_000
    progress = tf.keras.utils.Progbar(N)
    ppl = PPL(generator=gen)

    distances = []
    for i in range(N):
        progress.update(i)

        zs = ppl.get_latent_interpolations(space="w", z_dim=49, e=10e-2)

        d = ppl.compute_distance_sum(zs)

        distances.append(d)

    print()
    print("PPL Full:", np.mean(distances))

    # End PPL
    progress = tf.keras.utils.Progbar(N)
    distances = []
    for i in range(N):
        progress.update(i)

        zs = ppl.get_latent_interpolations(space="w", z_dim=49)

        d = ppl.compute_distance_sum([zs[0], zs[-1]])

        distances.append(d)

    print()
    print("PPL End:", np.mean(distances))


