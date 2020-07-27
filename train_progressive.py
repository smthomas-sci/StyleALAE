"""
Trains a full progressive model

Author: Simon Thomas
Date: 27/07/2020

"""

import argparse
import matplotlib.pyplot as plt
import os
import pickle
import logging
import datetime

from StyleALAE.models import *
from StyleALAE.optimizers import *
from StyleALAE.data import *
from StyleALAE.utils import ConfigParser, Summary

# ------------------------------------------ ARGUMENT PARSER --------------------------------------------------------- #
parser = argparse.ArgumentParser(description='Train progressive StyleALAE')
parser.add_argument('--config',
                    type=str,
                    default="/home/simon/PycharmProjects/StyleALAE/StyleALAE/configs/celeba_hq_256.config",
                    help='full path and filename to yaml config file')
args = parser.parse_args()
# -------------------------------------------------------------------------------------------------------------------- #

# CONFIG
config = ConfigParser(args.config)

# MODEL HYPER-PARAMETERS
X_DIM = 4  # NOTE: Always start from 4x4
Z_DIM = config.z_dim  # The size of the latent vector
F_N_LAYERS = config.F_layers  # The number of layers in the F (mapping) network
D_N_LAYERS = config.D_layers  # The number of layers in the D (discriminator) network
BASE_FEATURES = config.base_features  # The number of features in the 4x4 base feature space
LEVELS = config.levels  # The number of levels in training e.g. len([4, 8, 16, 32, 64]) == 5
BATCH_SIZES = config.batch_sizes  # The batch size for each level
ALPHA = config.alpha  # The learning rate - recommend 0.0015-0.003
GAMMAS = config.gammas  # The Gradient-Penalty gamma values for each level [0.1, 0.1, 10, ...]
FILTERS = config.filters  # The convolutional filters for each level
N = config.n  # The number of images in data set
K_IMAGES = config.k_images  # The number of images to train per level (500K-800K)
EPOCHS = int(K_IMAGES / N)  # The number of epochs to train to reach K_IMAGES
BLOCK_TYPE = config.block_type  # Block use "AdaIn" or "ModDemod"

# INPUT & OUTPUT
DATA_DIR = config.data_dir  # The data directory containing only png/jpg/tif files
RUN_DIR = config.run_dir  # The directory containing all output for particular run
LOG_DIR = os.path.join(RUN_DIR, "logs/")
OUT_DIR = os.path.join(RUN_DIR, "output/")
WEIGHT_DIR = os.path.join(RUN_DIR, "weights/")
FID_DIR = os.path.join(RUN_DIR, "fid/")

# Build Directories
for PATH in [RUN_DIR, LOG_DIR, OUT_DIR, WEIGHT_DIR, FID_DIR]:
    if not os.path.exists(PATH):
        os.mkdir(PATH)
        print(f"Creating dir: {PATH}")
        if "fid" in PATH:
            os.mkdir(os.path.join(PATH, "real/"))
            os.mkdir(os.path.join(PATH, "fake/"))
        else:
            # create run folders for each level
            for level in range(1, LEVELS + 1):
                dim = 2 ** (level + 1)
                os.mkdir(os.path.join(PATH, f"{dim}x{dim}/"))

# BEGIN TRAINING

# Remove stupid warnings --------------- #
tf.get_logger().setLevel(logging.ERROR)
# -------------------------------------- #

for level in range(1, LEVELS + 1):

    # TIME RUN
    start = datetime.datetime.now()

    # Show Level parameters
    X_DIM = 2 ** (level + 1)
    print(f"Level: {level}",
          f"DIM: {X_DIM}",
          f"Batch Size: {BATCH_SIZES[level - 1]}",
          f"GAMMA: {GAMMAS[level - 1]}", sep="\t\t")

    # Create dataset
    data_gen = create_data_set(data_directory=DATA_DIR, img_dim=X_DIM, batch_size=BATCH_SIZES[level - 1])

    # Get test data - is the same for each run!
    tf.random.set_seed(1234)
    test_z = tf.random.normal((16, Z_DIM), seed=1)
    test_batch = get_test_batch(data_gen)

    # ------------------ BUILD MODEL ------------------- #
    K.clear_session()
    # MODELS
    F = build_F(F_N_LAYERS, Z_DIM)
    G = build_base_generator(z_dim=Z_DIM, base_features=BASE_FEATURES, block_type=BLOCK_TYPE)
    E = build_base_encoder(z_dim=Z_DIM, filters=FILTERS[1])
    D = build_D(D_N_LAYERS, Z_DIM)

    # Expand models as needed
    models = []
    if level == 1:
        # Build composite model
        alae = ALAE(x_dim=X_DIM,
                    z_dim=Z_DIM,
                    f_model=F,
                    g_model=G,
                    e_model=E,
                    d_model=D,
                    merge=False)

        # Optimizers
        Adam_D, Adam_G, Adam_R = create_optimizers(α=ALPHA, β1=0.0, β2=0.99)

        alae.compile(d_optimizer=Adam_D,
                     g_optimizer=Adam_G,
                     r_optimizer=Adam_R,
                     γ=GAMMAS[level - 1],
                     alpha_step=None)

        models.append(alae)

    else:  # level > 1
        WEIGHT_LEVEL = level - 1
        OLD_DIM = X_DIM // 2
        OLD_WEIGHT_DIR = os.path.join(WEIGHT_DIR, f"{OLD_DIM}x{OLD_DIM}")

        for b in range(2, level + 1):
            # Load weights if pre-trained
            if b - 1 == WEIGHT_LEVEL:
                F.load_weights(os.path.join(OLD_WEIGHT_DIR, f"F_{OLD_DIM}x{OLD_DIM}_weights.h5"))
                G.load_weights(os.path.join(OLD_WEIGHT_DIR, f"G_{OLD_DIM}x{OLD_DIM}_weights.h5"))
                E.load_weights(os.path.join(OLD_WEIGHT_DIR, f"E_{OLD_DIM}x{OLD_DIM}_weights.h5"))
                D.load_weights(os.path.join(OLD_WEIGHT_DIR, f"D_{OLD_DIM}x{OLD_DIM}_weights.h5"))

            G, G_m = expand_generator(old_model=G, block=b,
                                      filters=FILTERS[b][1], z_dim=Z_DIM,
                                      noise_dim=2 ** (b + 1), block_type=BLOCK_TYPE)

            E, E_m = expand_encoder(old_model=E,
                                    filters=FILTERS[b],
                                    block=b,
                                    z_dim=Z_DIM)
        # Expansion finished
        # Build merge model
        alae_m = ALAE(x_dim=X_DIM,
                      z_dim=Z_DIM,
                      f_model=F,
                      g_model=G_m,
                      e_model=E_m,
                      d_model=D,
                      merge=True)

        # Optimizers
        Adam_D, Adam_G, Adam_R = create_optimizers(α=ALPHA, β1=0.0, β2=0.99)

        alae_m.compile(d_optimizer=Adam_D,
                       g_optimizer=Adam_G,
                       r_optimizer=Adam_R,
                       γ=GAMMAS[level - 1],
                       alpha_step=1 / (EPOCHS * N))

        # Build straight model
        alae_s = ALAE(x_dim=X_DIM,
                      z_dim=Z_DIM,
                      f_model=F,
                      g_model=G,
                      e_model=E,
                      d_model=D,
                      merge=False)

        # Optimizers
        Adam_D, Adam_G, Adam_R = create_optimizers(α=ALPHA, β1=0.0, β2=0.99)

        alae_s.compile(d_optimizer=Adam_D,
                       g_optimizer=Adam_G,
                       r_optimizer=Adam_R,
                       γ=GAMMAS[level - 1],
                       alpha_step=None)

        models.extend([alae_m, alae_s])

    # Train models
    for m, alae in enumerate(models):
        if len(models) == 1:
            print("training base model")
            # CALL BACK
            callbacks = [
                Summary(log_dir=os.path.join(LOG_DIR, f"{X_DIM}x{X_DIM}/"),
                        write_graph=False,
                        update_freq=50,  # every n batches
                        test_z=test_z,
                        test_batch=test_batch,
                        img_dir=os.path.join(OUT_DIR, f"{X_DIM}x{X_DIM}/"),
                        n=16,
                        weight_dir=os.path.join(WEIGHT_DIR, f"{X_DIM}x{X_DIM}/"),
                        )
            ]

        # MERGE + STRAIGHT
        if len(models) == 2:
            m = "straight" if m else "merge"
            print(f"training {m} model")

            # TRAINING
            callbacks = [
                Summary(log_dir=os.path.join(LOG_DIR, f"{X_DIM}x{X_DIM}/{m}"),
                        write_graph=False,
                        update_freq=50,  # every n batches
                        test_z=test_z,
                        test_batch=test_batch,
                        img_dir=os.path.join(OUT_DIR, f"{X_DIM}x{X_DIM}/"),
                        n=16,
                        weight_dir=os.path.join(WEIGHT_DIR, f"{X_DIM}x{X_DIM}/"),
                        )
            ]

        # FIT
        alae.fit(x=data_gen,
                 steps_per_epoch=N // BATCH_SIZES[level-1],
                 epochs=EPOCHS,
                 callbacks=callbacks
                 )
    # LEVEL COMPLETE
    end = datetime.datetime.now()
    print(f"{X_DIM}x{X_DIM} epoch running time:", end - start)

print("Training Complete.")