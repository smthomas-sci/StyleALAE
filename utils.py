"""

Various functions used in training

Author: Simon Thomas
Date: Jul-24-2020

"""
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.io as io
from skimage.transform import resize

from scipy.linalg import sqrtm

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

class Summary(TensorBoard):
    """
    A standard tensorboard call back but includes
    a visualisation call at the end of each epoch
    """
    def __init__(self, test_z, test_batch, img_dir, n, weight_dir, **kwargs):
        """
        :param log_dir - the log directory (including the run name),
        :param write_graph - include the graph?
        :param update_freq - "epoch" "batch" or int(n) batches
        :param test_z: the z vector to test on
        :param test_batch: the x, noise and constant vectors to test on
        :param out_dir: where to save the images (path)
        :param n: how many images to show
        :param weight_dir: where to save the weights (every epoch)
        :param kwargs: kwargs relating to TensorBoard class
        """
        super(Summary, self).__init__(**kwargs)
        self.test_z = test_z
        self.test_batch = test_batch
        self.img_dir = img_dir
        self.n = n
        self.weight_dir = weight_dir

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        print()
        self.visualise_progress(self.test_z, self.test_batch, epoch, self.n)

        self.save_weights(epoch)

    def visualise_progress(self, z_test, test_batch, epoch, n):
        # Generate Random Samples
        samples = self.model.generator([z_test[:n]] + [test_batch[1][:n]] + [test_batch[2][:n]]).numpy()
        # Reconstruction Inputs
        recons = self.model.inference(test_batch).numpy()

        samples = np.clip(samples, 0, 1)
        recons = np.clip(recons, 0, 1)

        # Show progress
        fig, ax = plt.subplots(3, n, figsize=(n, 3))
        for i in range(n):
            ax[0, i].imshow(test_batch[0][i])
            ax[0, i].axis("off")
            ax[1, i].imshow(recons[i])
            ax[1, i].axis("off")
            ax[2, i].imshow(samples[i])
            ax[2, i].axis("off")

        plt.text(-0.8, 0.4, "Orig.", transform=ax[0, 0].transAxes)
        plt.text(-1, 0.4, "Recon.", transform=ax[1, 0].transAxes)
        plt.text(-1.1, 0.4, "Sample", transform=ax[2, 0].transAxes)

        plt.text(0, 1.2, f"{epoch:04d}", transform=ax[0, 0].transAxes)

        dim = test_batch[0].shape[1]

        if self.model.merge:
            suffix = "_merge"
        else:
            suffix = ""

        fname = os.path.join(self.img_dir, f"progress_{dim:04d}{suffix}_{epoch:04d}.jpg")
        plt.savefig(fname)
        print("Plot saved at", fname)
        plt.close()

    def save_weights(self, epoch):
        if self.model.merge:
            suffix = "_merge_weights"
        else:
            suffix = "_weights"

        # Save weights and losses
        DIM = self.model.x_dim
        self.model.G.save_weights(f"{self.weight_dir}/G_{DIM}x{DIM}{suffix}_{epoch}" + ".h5")
        self.model.E.save_weights(f"{self.weight_dir}/E_{DIM}x{DIM}{suffix}_{epoch}" + ".h5")
        self.model.F.save_weights(f"{self.weight_dir}/F_{DIM}x{DIM}{suffix}_{epoch}" + ".h5")
        self.model.D.save_weights(f"{self.weight_dir}/D_{DIM}x{DIM}{suffix}_{epoch}" + ".h5")
        print("Weights Saved.")


def _generator(path, stop):
    files = [os.path.join(path, file) for file in os.listdir(path)][0:stop]
    i = 0
    while i < stop:
        i += 1
        file = files[i - 1]
        img = io.imread(file)
        img = resize(img, (229, 299), preserve_range=True).astype('float32')
        img = preprocess_input(img)
        yield np.expand_dims(img, 0)

class FID(object):
    """
    Calculate the FID score between two image distributions.

    Inception V3 has 23M parameters and so likely won't fit in memory along side
    the generative model. Ideally, generated images are saved to disk and FID
    is run externally.

    The workflow should be as follows:

    1. Save k real images to disk
    2. Run real images through InceptionV3 and create baseline score / save features
       >>> fid = FID(real_path=REAL_DIR, fake_path=FAKE_DIR, k=10000, baseline=None)
       >>> fid.get_real_features()
       >>> fid.create_baseline()
       >>> fid.save_real_features(fname)
    4. For each epoch at this image level e.g. 64x64, save k fake images to disk
       >>> fid.get_fake_features()
       >>> score = fid.score()
    """
    def __init__(self, real_path, fake_path, k=10000):
        """
        Build a self-container model and scorer for FID.
        :param real_path: the path to directory of real images
        :param fake_path: the path to directory of fake images
        :param k: how many images to use to calculate score
        :param baseline:
        """
        self.k = k
        self.real_gen = _generator(path=real_path, stop=self.k)
        self.fake_gen = _generator(path=fake_path, stop=self.k)
        self.model = InceptionV3(include_top=False,
                                weights="imagenet",
                                input_shape=(299, 299, 3),
                                pooling="avg")

    def get_real_features(self, path=None):
        print("Loading real features...")
        if path:
            self.real = np.load(path)
        else:
            self.real = self.model.predict(self.real_gen, steps=self.k, verbose=1)

    def get_fake_features(self):
        print("Loading fake features...")
        self.fake = self.model.predict(self.fake_gen, steps=self.k, verbose=1)

    def save_real_features(self, fname):
        print("saving real features...")
        np.save(fname, self.real)

    def calculate_fid(self, feat1, feat2):
        # calculate mean and covariance statistics
        mu1, sigma1 = np.mean(feat1, axis=0), np.cov(feat1, rowvar=False)
        mu2, sigma2 = np.mean(feat2, axis=0), np.cov(feat2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = np.sum(np.square(mu1 - mu2))
        # calculate sqrt of product between cov - r.dot(r) = x & r = sqrt(x)
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if np.iscomplexobj(covmean):
            # keeps only the real components (numpy)
            covmean = covmean.real
        # calculate score
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def score(self):
        return self.calculate_fid(self.real, self.fake)


if __name__ == "__main__":
    import timeit
    # start timer
    start = timeit.default_timer()

    REAL_DIR = "/home/simon/PycharmProjects/StyleALAE/data/celeba-128"
    FAKE_DIR = "/home/simon/Documents/Programming/Data/HistoPatches/Validation_256_labelled/"

    fid = FID(real_path=REAL_DIR, fake_path=FAKE_DIR, k=100)

    # Predict real features and calculate baseline
    fid.get_real_features()

    # Predict fake features and then calculate fid
    fid.get_fake_features()

    score = fid.score()
    print("FID:", score)

    # End of timer
    stop = timeit.default_timer()

    print('Time: ', stop - start)





