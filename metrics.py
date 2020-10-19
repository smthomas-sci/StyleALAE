"""
Metrics for determining the quality of generated and reconstructed images.

Author: Simon Thomas
Date: 28-07-2020
Updated: 13-10-2020

"""
import numpy as np
import os
import tensorflow as tf

from StyleALAE.data import *

# FID
# from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_iv3
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

class FID(object):
    """
    Calculate the FID score between two image distributions.

    Inception V3 has 23M parameters and so likely won't fit in memory along side
    the generative model. Ideally, generated images are saved to disk and FID
    is run externally.

    The workflow should be as follows:

    1. Save k real images to disk
    2. Run real images through InceptionV3 and create baseline score / save features
       >>> fid = FID(real_path=REAL_DIR, fake_path=FAKE_DIR, k=10000)
       >>> fid.get_real_features()
       >>> fid.save_real_features(fname)
    4. For each epoch at this image level e.g. 64x64, save k fake images to disk
       >>> fid.get_fake_features()
       >>> score = fid.score()


    Weights are NOT from tf.keras.applications.inceptionv3. Rather, it
    is taken from

    The images are load in range [0, 1], and so are preprocessed according
    to https://github.com/tensorflow/models/blob/33c61588cabda85598fbfbdce0d0329bbe42b4a4/research/slim/preprocessing/inception_preprocessing.py#L304

    Very close to the original which shares the same weights, adapted from
    https://tfhub.dev/tensorflow/tfgan/eval/inception/1'

    """
    def __init__(self, real_path, fake_path, k=10000, batch_size=10):
        """
        Build a self-container model and scorer for FID.
        :param real_path: the path to directory of real images
        :param fake_path: the path to directory of fake images
        :param k: how many images to use to calculate score
        """
        self.k = k
        self.batch_size = batch_size
        self.n_batches = self.k // self.batch_size
        self.real_gen = create_data_set(data_directory=real_path, img_dim=299, batch_size=self.batch_size, style=False)
        self.fake_gen = create_data_set(data_directory=fake_path, img_dim=299, batch_size=self.batch_size, style=False)
        self.model = tfhub.load("./StyleALAE/weights/inception_v3")

    def get_features(self, images):
        output = self.model(images)["pool_3"]
        return tf.reshape(output, (images.shape[0], 2048))

    def get_real_features(self, path=None):
        print("Loading real features...")
        if path:
            self.real = np.load(path)
        else:
            progress = tf.keras.utils.Progbar(self.n_batches)
            self.real = np.empty((self.k, 2048))
            for i, image in enumerate(self.real_gen):
                image = tf.subtract(image, 0.5)
                image = tf.multiply(image, 2.0)
                self.real[i*self.batch_size:i*self.batch_size+self.batch_size] = self.get_features(image)
                progress.update(i)
                if i == self.n_batches:
                    break
            progress.update(i, finalize=True)

    def get_fake_features(self):
        print("Loading fake features...")
        progress = tf.keras.utils.Progbar(self.n_batches)
        self.fake = np.empty((self.k, 2048))
        for i, image in enumerate(self.fake_gen):
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
            self.fake[i*self.batch_size:i*self.batch_size+self.batch_size] = self.get_features(image)
            progress.update(i)
            if i == self.n_batches:
                break
        progress.update(i, finalize=True)

    def save_real_features(self, fname):
        print("saving real features...")
        np.save(fname, self.real)

    def calculate_fid(self, feat1, feat2):
        # calculate mean and covariance statistics
        mu1, sigma1 = np.mean(feat1, axis=0), np.cov(feat1, rowvar=False)
        mu2, sigma2 = np.mean(feat2, axis=0), np.cov(feat2, rowvar=False)

        diff = mu1 - mu2

#         # CUSTOM
#         # calculate sum squared difference between means
#         ssdiff = np.sum(np.square(mu1 - mu2))
#         # calculate sqrt of product between cov - r.dot(r) = x & r = sqrt(x)
#         covmean = sqrtm(sigma1.dot(sigma2))
#         # check and correct imaginary numbers from sqrt
#         if np.iscomplexobj(covmean):
#             # keeps only the real components (numpy)
#             covmean = covmean.real
#         # calculate score
#         fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
#         return fid

        # PAPER
        # product might be almost singular
        covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
            warnings.warn(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


    def score(self):
        return self.calculate_fid(self.real, self.fake)


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

    N = 4
    progress = tf.keras.utils.Progbar(N)
    ppl = PPL(generator=gen)


    distances = []
    for i in range(N):
        progress.update(i)

        zs = ppl.get_latent_interpolations(space="z", z_dim=49, e=0.1)

        d = ppl.compute_distance_sum(zs)

        distances.append(d)

    print()
    print("PPL Full:", np.mean(distances))

    # End PPL
    progress = tf.keras.utils.Progbar(N)
    distances = []
    for i in range(N):
        progress.update(i)

        zs = ppl.get_latent_interpolations(space="z", z_dim=49)

        d = ppl.compute_distance_sum([zs[0], zs[-1]])

        distances.append(d)

    print()
    print("PPL End:", np.mean(distances))


