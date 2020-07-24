"""
A simple training script for expanding the network

Examples for filter size and batch size
for each level can be found in the CONFIG
files for ALAE as well as PINONEER (https://github.com/AaltoVision/pioneer/blob/master/src/train.py)

#                    4       8       16       32       64       128        256       512       1024
  LOD_2_BATCH_8GPU: [512,    256,     128,      64,      32,       32,        32,       32,        24]
  LOD_2_BATCH_4GPU: [512,    256,     128,      64,      32,       32,        32,       32,        16]
  LOD_2_BATCH_2GPU: [128,    128,     128,      64,      32,       32,        16]
  LOD_2_BATCH_1GPU: [128,    128,     128,      64,      32,       16]


"""

import matplotlib.pyplot as plt
import os
import pickle

from StyleALAE.models import *
from StyleALAE.optimizers import *
from StyleALAE.data import *
from StyleALAE.utils import Summary

import logging
tf.get_logger().setLevel(logging.ERROR)


# PARAMETERS
LEVELS = 6  # > 1
WEIGHT_LEVEL = 5
X_DIM = 2**(LEVELS+1)
Z_DIM = 100
F_N_LAYERS = 3
D_N_LAYERS = 3
BASE_FEATURES = 128
BATCH_SIZE = 32
ALPHA_STEP = None
DATA_DIR = "/home/simon/PycharmProjects/StyleALAE/data/celeba-128"
RUN_NAME = f"{X_DIM}x{X_DIM}_1"  #{np.random.randint(1, 100)}"
LOG_DIR = "/home/simon/PycharmProjects/StyleALAE/logs/" + RUN_NAME
IMG_DIR = "/home/simon/PycharmProjects/StyleALAE/output/" + RUN_NAME
WEIGHT_DIR = "/home/simon/PycharmProjects/StyleALAE/weights/" + RUN_NAME
OLD_WEIGHT_DIR = f"/home/simon/PycharmProjects/StyleALAE/weights/{X_DIM//2}x{X_DIM//2}_1"
N = None

FILTERS = {
    # E_in | E/G Filters
    1: [128, 128],
    2: [64, 128],
    3: [64, 64],
    4: [32, 64],
    5: [32, 32],
    6: [16, 32],
}

# PRE-RUN CHECKS
for PATH in [LOG_DIR, IMG_DIR, WEIGHT_DIR]:
    if not os.path.exists(PATH):
        os.mkdir(PATH)

# DATA
data_gen = create_data_set(data_directory=DATA_DIR, img_dim=X_DIM, batch_size=BATCH_SIZE)
if not N:  # this may be known in advance
    N = sum(1 for _ in data_gen)
EPOCHS = int(500000/(N*BATCH_SIZE)+1)

# --- MULTI-GPU TRAINING --- #
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
# Everything that creates variables should be under the strategy scope.
with strategy.scope():
    # MODELS
    F = build_F(F_N_LAYERS, Z_DIM)
    G = build_base_generator(z_dim=Z_DIM, base_features=BASE_FEATURES, block_type="AdaIN")
    E = build_base_encoder(z_dim=Z_DIM, filters=FILTERS[1])
    D = build_D(D_N_LAYERS, Z_DIM)

    # Expand models as needed
    for b in range(2, LEVELS+1):
        # Load weights if pre-trained
        if b-1 == WEIGHT_LEVEL:
            DIM = 2**(WEIGHT_LEVEL+1)
            F.load_weights(os.path.join(OLD_WEIGHT_DIR, f"F_{DIM}x{DIM}_weights_16.h5"))
            G.load_weights(os.path.join(OLD_WEIGHT_DIR, f"G_{DIM}x{DIM}_weights_16.h5"))
            E.load_weights(os.path.join(OLD_WEIGHT_DIR, f"E_{DIM}x{DIM}_weights_16.h5"))
            D.load_weights(os.path.join(OLD_WEIGHT_DIR, f"D_{DIM}x{DIM}_weights_16.h5"))

        G, G_m = expand_generator(old_model=G, block=b,
                                  filters=FILTERS[b][1], z_dim=Z_DIM,
                                  noise_dim=2**(b+1), block_type="AdaIN")

        E, E_m = expand_encoder(old_model=E,
                                  filters=FILTERS[b],
                                  block=b,
                                  z_dim=Z_DIM)

    # Build merge models
    alae_m = ALAE(x_dim=X_DIM,
                z_dim=Z_DIM,
                f_model=F,
                g_model=G_m,
                e_model=E_m,
                d_model=D,
                merge=True)

    # Optimizers
    Adam_D, Adam_G, Adam_R = create_optimizers(α=0.002, β1=0.0, β2=0.99)

    alae_m.compile(d_optimizer=Adam_D,
                 g_optimizer=Adam_G,
                 r_optimizer=Adam_R,
                 γ=10,
                 alpha_step=1/(EPOCHS*N))

    # Build straight models
    alae_s = ALAE(x_dim=X_DIM,
                z_dim=Z_DIM,
                f_model=F,
                g_model=G,
                e_model=E,
                d_model=D,
                merge=False)

    # Optimizers
    Adam_D, Adam_G, Adam_R = create_optimizers(α=0.002, β1=0.0, β2=0.99)

    alae_s.compile(d_optimizer=Adam_D,
                   g_optimizer=Adam_G,
                   r_optimizer=Adam_R,
                   γ=10,
                   alpha_step=None)




test_z = tf.random.normal((16, Z_DIM))
test_batch = get_test_batch(data_gen)

# TRAINING
callbacks = [
    Summary(log_dir=os.path.join(LOG_DIR, "merge"),
            write_graph=False,
            update_freq=50,  # every n batches
            test_z=test_z,
            test_batch=test_batch,
            img_dir=IMG_DIR,
            n=16,
            weight_dir=WEIGHT_DIR
                )
]

print("training merge model")
history_merge = alae_m.fit(x=data_gen,
                           steps_per_epoch=N,
                           epochs=EPOCHS,
                           callbacks=callbacks,
                           initial_epoch=0
                           )

print("training straight model")
history_straight = alae_s.fit(x=data_gen,
                              steps_per_epoch=N,
                              epochs=EPOCHS,
                              callbacks=callbacks,
                              initial_epoch=0
                              )









