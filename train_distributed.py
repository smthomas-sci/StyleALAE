"""
A simple training script


Multi-GPU
-  https://blog.paperspace.com/tensorflow-2-0-in-practice/
- https://stackoverflow.com/questions/62349329/distributed-training-using-mirrorstrategy-in-tensorflow-2-2-with-custom-training
- https://www.tensorflow.org/tutorials/distribute/custom_training


"""
"""
A simple training script

"""

import matplotlib.pyplot as plt
import os
import pickle

from StyleALAE.models import *
from StyleALAE.optimizers import *
from StyleALAE.data import *
from StyleALAE.utils import Summary

os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = str(6)


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

# DATA
#data_gen = create_data_set(data_directory=DATA_DIR, img_dim=4, batch_size=128)
data_gen = create_data_set(file_names=[os.path.join(DATA_DIR, file) for file in os.listdir(DATA_DIR)][0:128*20],
                           img_dim=4, batch_size=128)

if not N:  # this may be known in advance
    N = sum(1 for _ in data_gen)

EPOCHS = int(500000/(N*BATCH_SIZE)+1)

# Prepare the dataset. We use both the training & test MNIST digits.
# (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
# x_train = x_train.repeat(3, axis=-1)
# x_test = x_test.repeat(3, axis=-1)
# all_digits = np.concatenate([x_train, x_test])
# all_digits = all_digits.astype("float32") / 255.0
# all_digits = np.reshape(all_digits, (-1, 28, 28, 3))
#
# AUTOTUNE = tf.data.experimental.AUTOTUNE
#
# n = all_digits.shape[0]
# BATCH_SIZE_PER_REPLICA = 128
#
# def resize_tensor(tensor):
#     image = tf.image.resize(tensor, [4, 4])
#     noise_image = tf.random.normal([4, 4], mean=0, stddev=0.1)
#     constant = tf.ones([1])
#     return image, noise_image, constant
#
# dataset = tf.data.Dataset.from_tensor_slices(all_digits)
# dataset = dataset.map(resize_tensor, num_parallel_calls=AUTOTUNE)
# dataset = dataset.shuffle(buffer_size=n, seed=1234, reshuffle_each_iteration=True)
# dataset = dataset.batch(BATCH_SIZE_PER_REPLICA*2, drop_remainder=True)
# data_gen = dataset.prefetch(buffer_size=AUTOTUNE)
#
# EPOCHS = 3
#
# for img in data_gen:
#
#     X, c, n = img
#
#     print(X.shape)
#
#     break


# --- MULTI-GPU TRAINING --- #
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
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



# CALLBACKS
test_z = tf.random.normal((16, Z_DIM))
test_batch = get_test_batch(data_gen)

callbacks = [
    Summary(log_dir=LOG_DIR,
            write_graph=False,
            update_freq=50,  # every n batches
            test_z=test_z,
            test_batch=test_batch,
            img_dir=IMG_DIR,
            n=16,
            weight_dir=WEIGHT_DIR
            )
]

# TRAINING
history = alae.fit(x=data_gen,
                   steps_per_epoch=N,
                   epochs=EPOCHS,
                   callbacks=callbacks
                   )








