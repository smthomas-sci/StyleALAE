"""
A simple training script


Multi-GPU
-  https://blog.paperspace.com/tensorflow-2-0-in-practice/
- https://stackoverflow.com/questions/62349329/distributed-training-using-mirrorstrategy-in-tensorflow-2-2-with-custom-training

"""
from keras.utils import Progbar

import matplotlib.pyplot as plt
import os
import pickle

from StyleALAE.models import *
from StyleALAE.optimizers import *
from StyleALAE.data import *
from StyleALAE.utils import Summary

# PARAMETERS
X_DIM = 4
Z_DIM = 100
F_N_LAYERS = 3
D_N_LAYERS = 3
BASE_FEATURES = 128
BATCH_SIZE = 128
ALPHA_STEP = None
DATA_DIR = "/home/simon/PycharmProjects/StyleALAE/data/celeba-128"
RUN_NAME = f"{X_DIM}x{X_DIM}_1"  #{np.random.randint(1, 100)}"
LOG_DIR = "/home/simon/PycharmProjects/StyleALAE/logs/" + RUN_NAME
IMG_DIR = "/home/simon/PycharmProjects/StyleALAE/output/" + RUN_NAME
WEIGHT_DIR = "/home/simon/PycharmProjects/StyleALAE/weights/" + RUN_NAME
N = None

# PRE-RUN CHECKS
for PATH in [LOG_DIR, IMG_DIR, WEIGHT_DIR]:
    if not os.path.exists(PATH):
        os.mkdir(PATH)



# --- MULTI-GPU TRAINING --- #
strategy = tf.distribute.MirroredStrategy()
N_GPUS = strategy.num_replicas_in_sync
print('Number of devices: {}'.format(N_GPUS))
# Everything that creates variables should be under the strategy scope.
with strategy.scope():
    # MODELS
    F = build_F(F_N_LAYERS, Z_DIM)
    G = build_base_generator(z_dim=Z_DIM, base_features=BASE_FEATURES, block_type="AdaIN")
    E = build_base_encoder(z_dim=Z_DIM, filters=[128, 128])
    D = build_D(D_N_LAYERS, Z_DIM)

    # Build composite model
    alae = ALAE(x_dim=X_DIM,
                z_dim=Z_DIM,
                f_model=F,
                g_model=G,
                e_model=E,
                d_model=D,
                merge=False)

    # Optimizers
    Adam_D, Adam_G, Adam_R = create_optimizers(α=0.002, β1=0.0, β2=0.99)

    alae.compile(d_optimizer=Adam_D,
                 g_optimizer=Adam_G,
                 r_optimizer=Adam_R,
                 γ=0.1,
                 alpha_step=None)


    @tf.function
    def discriminator_train_step(batch):
        # Step I - Update Discriminator  #
        # -------------------------------#

        # Random mini-batch from data set
        x_real, noise, constant = batch

        # samples from prior N(0, 1)
        z = K.random_normal((BATCH_SIZE, alae.z_dim))
        # generate fake images
        x_fake = alae.generator([z, noise, constant])

        # Compute loss and apply gradients
        with tf.GradientTape() as tape:

            fake_pred = alae.discriminator(x_fake)

            real_pred = alae.discriminator(x_real)

            loss_d = discriminator_logistic_non_saturating(real_pred, fake_pred)

            # Add the R1 term
            if alae.γ > 0:

                x_real = alae.real_as_tensor(x_real) #tf.Variable(x_real, dtype=tf.float32)
                with tf.GradientTape() as r1_tape:
                    r1_tape.watch(x_real)
                    # 1. Get the discriminator output for real images
                    pred = alae.discriminator(x_real)

                # 2. Calculate the gradients w.r.t to the real images.
                grads = r1_tape.gradient(pred, [x_real])[0]

                # 3. Calculate the squared norm of the gradients
                r1_penalty = tf.nn.compute_average_loss(K.sum(K.square(grads), axis=[1, 2, 3]))
                loss_d += alae.γ / 2 * r1_penalty

        gradients = tape.gradient(loss_d, alae.θ_E + alae.θ_D)
        alae.optimizer_d.apply_gradients(zip(gradients, alae.θ_E + alae.θ_D))

        return loss_d


    @tf.function
    def generator_train_step(batch):
        # ----------------------------#
        #  Step II - Update Generator #
        # ----------------------------#
        x_real, noise, constant = batch
        # samples from prior N(0, 1)
        z = K.random_normal((BATCH_SIZE, alae.z_dim))
        # Compute loss and apply gradients
        with tf.GradientTape() as tape:
            fake_pred = alae.discriminator(alae.generator([z, noise, constant]))

            loss_g = generator_logistic_non_saturating(fake_pred)

        gradients = tape.gradient(loss_g, alae.θ_F + alae.θ_G)
        alae.optimizer_g.apply_gradients(zip(gradients, alae.θ_F + alae.θ_G))

        return loss_g


    @tf.function
    def reciprocal_train_step(batch):
        # ------------------------------#
        #  Step III - Update Reciprocal #
        # ------------------------------#
        x_real, noise, constant = batch
        # samples from prior N(0, 1)
        z = K.random_normal((BATCH_SIZE, alae.z_dim))
        # Get w
        w = alae.F(z)
        # Compute loss and apply gradients
        with tf.GradientTape() as tape:
            w_pred = alae.reciprocal([w, noise, constant])

            loss_r = l2(w, w_pred)

        gradients = tape.gradient(loss_r, alae.θ_G + alae.θ_E)
        alae.optimizer_r.apply_gradients(zip(gradients, alae.θ_G + alae.θ_E))

        return loss_r


# DATA
data_gen = create_data_set(data_directory=DATA_DIR, img_dim=4, batch_size=BATCH_SIZE*N_GPUS)
if not N:  # this may be known in advance
    N = sum(1 for _ in data_gen)

EPOCHS = int(500000/(N*BATCH_SIZE)+1)

# Distribute data set
data_gen = strategy.experimental_distribute_dataset(data_gen)


@tf.function
def distributed_train(batch):

    d_loss = strategy.run(discriminator_train_step, args=(batch,))
    g_loss = strategy.run(generator_train_step, args=(batch,))
    r_loss = strategy.run(reciprocal_train_step, args=(batch,))

    d_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, d_loss, axis=None)
    g_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, g_loss, axis=None)
    r_loss = strategy.reduce(tf.distribute.ReduceOp.MEAN, r_loss, axis=None)

    return d_loss, g_loss, r_loss


# also change train() to use distributed_train instead of train_step
def train(dataset, epochs):
    for epoch in range(epochs):
        print(epoch, "of", epochs)
        for i, image_batch in enumerate(dataset):
            d_loss, g_loss, r_loss = distributed_train(image_batch)
            if i % 10 == 0:
                print(d_loss.numpy(), g_loss.numpy(), r_loss.numpy())

train(data_gen, EPOCHS)

