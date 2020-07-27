"""
Contains the basic of loading optimizers
"""

import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

def create_optimizers(α=0.002, β1=0.0, β2=0.99, ε=K.epsilon()):
    """
    Create n adam optimizers with given hyper-paramaters
    :param α: alpha / learning rate, default=0.002, paper range (0.0015 - 0.003)
    :param β1: beta1 for adam optimizer, default = 0.0
    :param β2: beta2 for adam optimizer, default = 0.99
    :param ε: epsilon for avoiding division errors.
    :param decay: the averaging decay for the generator Exponential Moving Average
    :return: [d_opt, g_opt, r_opt]
    """
    d_opt = Adam(α, β1, β2, ε)
    g_opt = Adam(α, β1, β2, ε)
    r_opt = Adam(α, β1, β2, ε)
    return [d_opt, g_opt, r_opt]
