"""
A simple training script


Multi-GPU
-  https://blog.paperspace.com/tensorflow-2-0-in-practice/
- https://stackoverflow.com/questions/62349329/distributed-training-using-mirrorstrategy-in-tensorflow-2-2-with-custom-training
- https://www.tensorflow.org/tutorials/distribute/custom_training


"""

import sys

sys.path.append("..")

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
# DATA_DIR = "/home/simon/PycharmProjects/StyleALAE/data/celeba-128"
# RUN_NAME = f"{X_DIM}x{X_DIM}_1"  #{np.random.randint(1, 100)}"
# LOG_DIR = "/home/simon/PycharmProjects/StyleALAE/logs/" + RUN_NAME
# IMG_DIR = "/home/simon/PycharmProjects/StyleALAE/output/" + RUN_NAME
# WEIGHT_DIR = "/home/simon/PycharmProjects/StyleALAE/weights/" + RUN_NAME
DATA_DIR = "/home/Student/s4200058/Dermo/celeba-256/"
RUN_NAME = f"{X_DIM}x{X_DIM}_1"
LOG_DIR = "/home/Student/s4200058/Dermo/logs/" + RUN_NAME
IMG_DIR = "/home/Student/s4200058/Dermo/output/" + RUN_NAME
WEIGHT_DIR = "/home/Student/s4200058/Dermo/weights/" + RUN_NAME

N = None

# PRE-RUN CHECKS
for PATH in [LOG_DIR, IMG_DIR, WEIGHT_DIR]:
    if not os.path.exists(PATH):
        os.mkdir(PATH)

# DATA
data_gen = create_data_set(data_directory=DATA_DIR, img_dim=4, batch_size=128)
if not N:  # this may be known in advance
    N = sum(1 for _ in data_gen)

EPOCHS = int(500000/(N*BATCH_SIZE)+1)



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

    loss_object = tf.keras.losses.BinaryCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE)


def train_step(inputs):
    batch = inputs
    batch_size = batch[0].shape[0]

    if not batch_size:
        batch_size = 1

    # Update alphas on Fade
    if alae.merge:
        alae.alpha += alae.alpha_step
        alae.G.get_layer("Fade_G").alpha.assign(alae.alpha)
        alae.E.get_layer("Fade_E").alpha.assign(alae.alpha)
        alae.E.get_layer("Fade_E_w").alpha.assign(alae.alpha)

    # -------------------------------#
    # Step I - Update Discriminator  #
    # -------------------------------#

    # Random mini-batch from data set
    x_real, noise, constant = batch

    # samples from prior N(0, 1)
    z = K.random_normal((batch_size, alae.z_dim))
    # generate fake images
    x_fake = alae.generator([z, noise, constant])

    # Compute loss and apply gradients
    with tf.GradientTape() as tape:

        fake_pred = alae.discriminator(x_fake)

        real_pred = alae.discriminator(x_real)

        loss_d = discriminator_logistic_non_saturating(real_pred, fake_pred, bs=BATCH_SIZE)

        # Add the R1 term
        if alae.γ > 0:
            with tf.GradientTape() as r1_tape:
                r1_tape.watch(x_real)
                # 1. Get the discriminator output for real images
                pred = alae.discriminator(x_real)

            # 2. Calculate the gradients w.r.t to the real images.
            grads = r1_tape.gradient(pred, [x_real])[0]

            # 3. Calculate the squared norm of the gradients
            r1_penalty = K.sum(K.square(grads))
            loss_d += alae.γ / 2 * r1_penalty

    gradients = tape.gradient(loss_d, alae.θ_E + alae.θ_D)
    alae.optimizer_d.apply_gradients(zip(gradients, alae.θ_E + alae.θ_D))

    # ----------------------------#
    #  Step II - Update Generator #
    # ----------------------------#

    # samples from prior N(0, 1)
    z = K.random_normal((batch_size, alae.z_dim))
    # Compute loss and apply gradients
    with tf.GradientTape() as tape:

        fake_pred = alae.discriminator(alae.generator([z, noise, constant]))

        loss_g = generator_logistic_non_saturating(fake_pred, bs=BATCH_SIZE)

    gradients = tape.gradient(loss_g, alae.θ_F + alae.θ_G)
    alae.optimizer_g.apply_gradients(zip(gradients, alae.θ_F + alae.θ_G))

    # ------------------------------#
    #  Step III - Update Reciprocal #
    # ------------------------------#

    # samples from prior N(0, 1)
    z = K.random_normal((batch_size, alae.z_dim))
    # Get w
    w = alae.F(z)
    # Compute loss and apply gradients
    with tf.GradientTape() as tape:

        w_pred = alae.reciprocal([w, noise, constant])

        loss_r = l2(w, w_pred)

    gradients = tape.gradient(loss_r, alae.θ_G + alae.θ_E)
    alae.optimizer_r.apply_gradients(zip(gradients, alae.θ_G + alae.θ_E))

    return {"loss_d": loss_d, "loss_g": loss_g, "loss_r": loss_r, "loss_gp": r1_penalty}

@tf.function
def distributed_train_step(dist_inputs):
  per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
  return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


dist_dataset = strategy.experimental_distribute_dataset(data_gen)


# TRAIN
for dist_inputs in dist_dataset:
  print(distributed_train_step(dist_inputs))