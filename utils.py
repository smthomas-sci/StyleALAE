"""

Various functions used in training

"""
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.callbacks import TensorBoard


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
