"""
Data tools. Includes both a keras Sequence

Author: Simon Thomas
Date: Jul-06-2020

"""
import numpy as np
import os
import tensorflow as tf

from tensorflow.keras.utils import Sequence
from skimage.transform import resize
from skimage import io

AUTOTUNE = tf.data.experimental.AUTOTUNE


def create_data_set(data_directory=None, file_names=None, img_dim=4, batch_size=12):
    """
    Creates a data set using the files in the data directory. If
    file_names is specified, only these file names will be used
    in the data set.
    :param data_directory: the path to the data directory.
    :param file_names: a list of full path file names
    :param img_dim: the size of the image
    :param batch_size: the batch size. Can be updated as ds = ds.batch(new_batch_size)
    :return: data set (ds)
    :return: n
    """
    if not file_names:
        file_names = [os.path.join(data_directory, file) for file in os.listdir(data_directory)]

    def parse_image(filename):
        """
        Reads the image and returns the image, noise and constant tensors
        :param filename: the image to load
        :return: style batch - ( image, noise_img, constant)
        """
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [img_dim, img_dim])
        noise_image = tf.random.normal([img_dim, img_dim], mean=0, stddev=1)
        constant = tf.ones([1])
        return image, noise_image, constant

    n = len(file_names)
    ds = tf.data.Dataset.from_tensor_slices(file_names)
    ds = ds.shuffle(buffer_size=n, seed=1234, reshuffle_each_iteration=True)
    ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def get_test_batch(data_set):
    """
    Returns the first batch from the data set
    :param data_set: data_set object
    :return: batch of tensors
    """
    for i, batch in enumerate(data_set):
        if i == 0:
            break
    return batch

class DataGenerator(Sequence):
    """
    A data generator that inherits the keras.utils.Sequence
    class so that it can be used with multi-processing
    """
    def __init__(self, directory, batch_size, img_dim=None, shuffle=True,
                 style=False, target_dim=None):
        """
        Inputs:
            directory - the directory where the images are
            batch_size - the batch size
            img_dim -  the size of the image if wanting to resize
            shuffle - whether to shuffle the images
            style - whether to include styleVAE noise as part of X
            target_dim - the target size in progressive growing eg. 4->8->...256, target is 256
        Outputs:
            if style:
                batch of images (batch_size, img_dim, img_dim, 3),
                noise (batch_size, target_dim, target_dim, 1),
                constant (batch_size, 1)
            else:
                batch of images, batch of images
        """
        self.dir = directory
        self.batch_size = batch_size
        self.img_dim = img_dim
        self.shuffle = shuffle
        self.style = style
        self.target_dim = target_dim
        # ---------------------------------------- #
        self.files = np.array(os.listdir(self.dir))
        self.n = len(self.files)
        self.indices = np.arange(self.n)
        if self.shuffle:
            np.random.seed(0)
            np.random.shuffle(self.indices)
        self.files_in_batch = []
        self.style = style
        self.pos = 0

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return self.n // self.batch_size

    def _indices_at(self, batch_index):
        """Returns the list of indices for batch index"""
        return self.indices[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indices of the batch
        indices = self._indices_at(index)

        # Get list of files in batch
        self.files_in_batch = self.files[indices]

        batch = []
        self.files_in_batch = []
        for i in indices:

            # Create filename
            fname = os.path.join(self.dir, self.files[i])

            # Load image & scale between 0-1
            img = io.imread(fname)[:, :, 0:3] / 255.

            # resize
            if img.shape[0] != self.img_dim:
                img = resize(img, (self.img_dim, self.img_dim))

            batch.append(img)

        batch = np.stack(batch)

        if self.style:

            X = [
                # Input Image
                batch,
                # Noise Image
                np.random.normal(0, 1, (self.batch_size, self.target_dim, self.target_dim, 1)),

                # Constant input
                np.ones((self.batch_size, 1, 1))
            ]
            return X

        # Standard
        else:
            return batch, batch

    def set_files(self, files):
        self.files = files
        self.n = len(self.files)
        self.shuffle = True
        self.indices = np.arange(self.n)
        print("setting files and resetting generator.")

    def __next__(self):
        if self.pos >= (self.n // self.batch_size):
            self.pos = -1
        result = self[self.pos]
        self.pos += 1
        return result


def create_patch_generator(filename, dim, scale=True):
    """
    Creates a multi-patch generator from a whole image. This makes
    processing whole images much more efficient.

    Note: It does NOT use overlapping patches, but rather pads the image so
          that even size patches can be sampled.
    """
    # Load Image
    print("reading in", filename, "...")
    image = io.imread(filename) / 255.

    # Scale to 2x reduction
    if scale:
        image = resize(image, (image.shape[0] // 2, image.shape[1] // 2))

    # Pad Image
    row_add = dim - (image.shape[0] % dim)
    col_add = dim - (image.shape[1] % dim)
    image_padded = np.pad(image, ([0, row_add], [0, col_add], [0, 0]), constant_values=[1])

    print("Converting to tensorflow dataset")
    # Convert to tensor
    image_padded_tensor = tf.convert_to_tensor(image_padded)

    def get_patch(index):
        r = index[0]
        c = index[1]
        return image_padded_tensor[r * dim:r * dim + dim, c * dim:c * dim + dim]

    # Get indices
    n_rows = image_padded.shape[0] // dim
    n_cols = image_padded.shape[1] // dim

    indices = []
    for r in range(n_rows):
        for c in range(n_cols):
            indices.append([r, c])
    indices = np.array(indices)

    # Create dataset
    ds = tf.data.Dataset.from_tensor_slices((indices))
    ds = ds.map(get_patch)
    ds = ds.batch(12)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds, (n_rows, n_cols)

if __name__ == "__main__":



    DATA_DIR = "/home/simon/PycharmProjects/StyleALAE/data/celeba-128"
    BATCH_SIZE = 12
    DIM = 4
    data_gen = DataGenerator(directory=DATA_DIR,
                             batch_size=BATCH_SIZE,
                             img_dim=DIM,
                             style=True,
                             target_dim=DIM)

    x, noise, constant = next(data_gen)

    files = [os.path.join(DATA_DIR, file) for file in os.listdir(DATA_DIR)]

    ds = create_data_set(file_names=files, img_dim=4, batch_size=128)

    print(ds)
    #N = sum(1 for _ in ds)
    for i, batch in enumerate(ds):
        if i == 0:
            break
    print(batch)
    # print("one")
    # for batch in ds:
    #     print(type(batch))
    # print("two")
    # for batch in ds:
    #     print(type(batch))
