"""
StyleALAE models / sub-models

Author: Simon Thomas
Date: Jul-06-2020

"""
from StyleALAE.layers import *
from StyleALAE.losses import *

import sys

def build_F(n_layers, z_dim, lrmul=0.01):
    """
    Builds the F network, mapping z to w.
    :param n_layers: the number of layers in the network
    :param z_dim: the size of the layers
    :param lrmul: the learning rate multiplier (2 order of magnitude as per StyleGAN 2)
    :return: F model
    """
    F_input = Input(shape=(z_dim,))
    x = F_input
    for i in range(n_layers):
        x = DenseEQ(units=z_dim, name=f"F_dense_{i+1}", lrmul=lrmul)(x)
        x = LeakyReLU(0.2, name=f"F_act_{i+1}")(x)
    F_output = x
    F = Model(inputs=[F_input], outputs=[F_output], name="F")
    return F


def build_D(n_layers, z_dim):
    """
    Builds the discriminator network.
    :param n_layers: the number of the layers
    :param z_dim: the size of the layers
    :return: D model
    """
    D_input = Input(shape=(z_dim,))
    x = D_input
    for i in range(n_layers):
        x = DenseEQ(units=z_dim, name=f"D_dense_{i+1}")(x)
        x = LeakyReLU(0.2, name=f"D_act_{i+1}")(x)
    D_output = DenseEQ(units=1)(x)
    D = Model(inputs=[D_input], outputs=[D_output], name="D")
    return D


def build_base_generator(z_dim, base_features, block_type="AdaIN"):
    """
    Builds the base generator network
    :param z_dim: the z-dimension / latent space
    :param base_features: the number of features at the base e.g. 4x4xbase_features
    :param block_type: which generator block type to use "AdaIN" or "ModDemod"
    :return: G model
    """

    # Set the block type
    GeneratorBlock = GeneratorBlockAdaIN if block_type == "AdaIN" else GeneratorBlockModDemod

    # Create inputs
    w_inputs = [Input(shape=(z_dim,), name="G_w_input_1")]
    noise_input = Input(shape=(4, 4, 1), name="G_noise_input")
    constant_input = Input(shape=(1, 1), name="G_constant_input")
    G_inputs = w_inputs + [noise_input, constant_input]

    # Constant Start - e.g. 4x4x512
    x = Dense(units=4*4*base_features, name=f"G_base")(constant_input)
    x = Reshape((4, 4, base_features), name=f"G_base_reshape")(x)

    # Create first style block
    block = 1
    style = G_inputs[block-1]
    x = GeneratorBlock(filters=base_features, block=block,
                       z_dim=z_dim, name=f"G_block_{block}_style")([x, noise_input, style])

    G_output = tRGB(block)(x)

    G = Model(inputs=[G_inputs], outputs=[G_output], name="G")

    return G


def build_base_encoder(z_dim, filters, dim=4):
    """
    Builds the base encoder model, by default starting with a 4x4x3 input
    :param z_dim: the size of the z-dimension / latent space to map to
    :param filters: list of filters for the block e.g. [input_filters, block_filters]
    :param dim: the input image size (4x4x3)
    :return: E (encoder model)
    """
    E_input = Input(shape=(dim, dim, 3), name="E_input")

    x, w1, w2 = EncoderBlockRouter(filters=[filters[0], filters[1]], block=1,
                                   z_dim=z_dim, name="encoder_block_router_1")(E_input)

    w = Add(name="E_Final_Sum_base")([w1, w2])

    E_output = w
    E = Model(inputs=[E_input], outputs=[E_output], name="E")
    return E


class ALAE(Model):
    """
    An Adversarial Latent Autoencoder Model (ALAE), self-contained for
    training at each block size.
    """
    def __init__(self, x_dim, z_dim, f_model, g_model, e_model, d_model, merge, style_mix_step=16):
        super(ALAE, self).__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.F = f_model
        self.G = g_model
        self.E = e_model
        self.D = d_model
        self.fade = merge
        self.levels = int(np.log2(self.x_dim/2))  # the number of blocks
        self.style_mix_step = style_mix_step

        # Composite models
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        self.reciprocal = self.build_reciprocal()
        self.inference = self.build_inference()
        # self.styleMixer = self.build_styleMixer() TO DO

    def build_discriminator(self):
        discriminator_in = Input(shape=(self.x_dim, self.x_dim, 3))
        discriminator_out = self.D(self.E(discriminator_in))
        return Model(inputs=[discriminator_in], outputs=[discriminator_out], name="discriminator")

    def build_generator(self):
        z_input = Input(shape=(self.z_dim,))
        noise_input = Input(shape=(self.x_dim, self.x_dim, 1))
        constant_input = Input(shape=(1, 1))
        generator_ins = [z_input, noise_input, constant_input]

        # Map z -> w
        w = self.F(z_input)
        x = self.G([w]*self.levels + [noise_input, constant_input])
        generator_out = x
        return Model(inputs=generator_ins, outputs=[generator_out], name="generator")

    def build_reciprocal(self):
        w_in = Input(shape=(self.z_dim,))  # W is input
        noise_input = Input(shape=(self.x_dim, self.x_dim, 1))
        constant_input = Input(shape=(1, 1))
        reciprocal_ins = [w_in, noise_input, constant_input]

        g_ins = [w_in]*self.levels + [noise_input, constant_input]
        reciprocal_out = self.E(self.G(g_ins))
        return Model(inputs=reciprocal_ins, outputs=[reciprocal_out], name="reciprocal")

    def build_inference(self):
        inference_in = Input(shape=(self.x_dim, self.x_dim, 3), name="inference_input")
        noise_input = Input(shape=(self.x_dim, self.x_dim, 1))
        constant_input = Input(shape=(1, 1))

        inference_ins = [inference_in, noise_input, constant_input]

        w = self.E(inference_in)
        g_ins = [w]*self.levels + [noise_input, constant_input]
        inference_out = self.G(g_ins)
        return Model(inputs=inference_ins, outputs=[inference_out], name="inference")

    def compile(self, d_optimizer, g_optimizer,  r_optimizer, γ=10, alpha_step=None):
        """
        Overrides the compile step. If it is a merge model,
        then the Fade layers are initialised and alphas are
        created.
        """
        super(ALAE, self).compile()
        self.optimizer_d = d_optimizer
        self.optimizer_g = g_optimizer
        self.optimizer_r = r_optimizer
        self.γ = γ

        if self.fade:
            self.alpha_step = alpha_step

        # get trainable params
        self.θ_F = self.F.trainable_weights
        self.θ_G = self.G.trainable_weights
        self.θ_E = self.E.trainable_weights
        self.θ_D = self.D.trainable_weights

        # Create loss trackers
        self.d_loss_tracker = tf.keras.metrics.Mean(name="loss_d")
        self.g_loss_tracker = tf.keras.metrics.Mean(name="loss_g")
        self.r_loss_tracker = tf.keras.metrics.Mean(name="loss_r")
        self.gp_loss_tracker = tf.keras.metrics.Mean(name="loss_r1")

        # Create internal step tracker
        self.step = 0

    def style_mixing_regularization(self, batch_size, noise, constant):
        # Reset step
        self.step *= 0

        # samples from prior N(0, 1)
        z1 = tf.random.normal(shape=(batch_size, self.z_dim))
        z2 = tf.random.normal(shape=(batch_size, self.z_dim))

        # Get ws
        w1 = self.F(z1)
        w2 = self.F(z2)

        # Get random position
        pos = tf.random.uniform(shape=[], minval=0, maxval=self.levels, dtype="int32")

        # Compute loss and apply gradients
        with tf.GradientTape() as tape:
            x = self.G([w1]*pos + [w2]*((self.levels-1) - pos) + [noise, constant])
            fake_pred = self.discriminator(x)
            loss_g_style_mix = generator_logistic_non_saturating(fake_pred, None)
        gradients = tape.gradient(loss_g_style_mix, self.θ_F + self.θ_G)
        self.optimizer_g.apply_gradients(zip(gradients, self.θ_F + self.θ_G))

    def train_step(self, batch):
        """
        Custom training step - follows algorithm of ALAE e.g. Step I,II & III
        :param batch:
        :return: losses
        """
        batch_size = batch[0].shape[0]

        if not batch_size:
            batch_size = 1

        # Update alphas on Fade
        if self.fade:
            self.G.get_layer("Fade_G").alpha.assign_add(self.alpha_step)
            self.E.get_layer("Fade_E").alpha.assign_add(self.alpha_step)
            self.E.get_layer("Fade_E_w").alpha.assign_add(self.alpha_step)

        # -------------------------------#
        # Step I - Update Discriminator  #
        # -------------------------------#

        # Random mini-batch from data set
        x_real, noise, constant = batch

        batch_size = tf.shape(x_real)[0]

        # samples from prior N(0, 1)
        z = tf.random.normal(shape=(batch_size, self.z_dim))
        # generate fake images
        x_fake = self.generator([z, noise, constant])

        # Compute loss and apply gradients
        with tf.GradientTape() as tape:

            fake_pred = self.discriminator(x_fake)

            real_pred = self.discriminator(x_real)

            loss_d = discriminator_logistic_non_saturating(real_pred, fake_pred)

            # Add the R1 term
            if self.γ > 0:

                with tf.GradientTape() as r1_tape:
                    r1_tape.watch(x_real)
                    # 1. Get the discriminator output for real images
                    pred = self.discriminator(x_real)

                # 2. Calculate the gradients w.r.t to the real images.
                grads = r1_tape.gradient(pred, [x_real])[0]

                # 3. Calculate the squared norm of the gradients
                r1_penalty = tf.reduce_sum(tf.square(grads))
                loss_d += self.γ / 2 * r1_penalty

        gradients = tape.gradient(loss_d, self.θ_E + self.θ_D)
        self.optimizer_d.apply_gradients(zip(gradients, self.θ_E + self.θ_D))

        # ----------------------------#
        #  Step II - Update Generator #
        # ----------------------------#

        # samples from prior N(0, 1)
        z = tf.random.normal(shape=(batch_size, self.z_dim))
        # Compute loss and apply gradients
        with tf.GradientTape() as tape:

            fake_pred = self.discriminator(self.generator([z, noise, constant]))

            loss_g = generator_logistic_non_saturating(fake_pred, None)

        gradients = tape.gradient(loss_g, self.θ_F + self.θ_G)
        self.optimizer_g.apply_gradients(zip(gradients, self.θ_F + self.θ_G))

        # -------  Style Mixing ------- #
        self.step += 1
        if self.step % self.style_mix_step == 0:
            print("Style mixxing...")
            self.style_mixing_regularization(batch_size, noise, constant)
        # ------------------------------#

        # ------------------------------#
        #  Step III - Update Reciprocal #
        # ------------------------------#

        # samples from prior N(0, 1)
        z = tf.random.normal(shape=(batch_size, self.z_dim))
        # Get w
        w = self.F(z)
        # Compute loss and apply gradients
        with tf.GradientTape() as tape:

            w_pred = self.reciprocal([w, noise, constant])

            loss_r = l2(w, w_pred)

        gradients = tape.gradient(loss_r, self.θ_G + self.θ_E)
        self.optimizer_r.apply_gradients(zip(gradients, self.θ_G + self.θ_E))

        if self.γ == 0:
            r1_penalty = 0.0
        # Update loss trackers
        self.d_loss_tracker.update_state(loss_d)
        self.g_loss_tracker.update_state(loss_g)
        self.r_loss_tracker.update_state(loss_r)
        self.gp_loss_tracker.update_state(r1_penalty)

        return {"loss_d": self.d_loss_tracker.result(),
                "loss_g": self.g_loss_tracker.result(),
                "loss_r": self.r_loss_tracker.result(),
                "loss_gp": self.gp_loss_tracker.result()}

    def call(self, inputs):
        return inputs

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker, self.r_loss_tracker, self.gp_loss_tracker]


