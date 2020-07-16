"""
StyleALAE models / sub-models

Author: Simon Thomas
Date: Jul-06-2020

"""
from StyleALAE.layers import *
from StyleALAE.losses import *


def build_F(n_layers, z_dim):
    """
    Builds the F network, mapping z to w.
    :param n_layers: the number of layers in the network
    :param z_dim: the size of the layers
    :return: F model
    """
    F_input = Input(shape=(z_dim,))
    x = F_input
    for i in range(n_layers):
        x = DenseEQ(units=z_dim, name=f"F_dense_{i+1}")(x)
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
    def __init__(self, x_dim, z_dim, f_model, g_model, e_model, d_model, merge):
        super(ALAE, self).__init__()
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.F = f_model
        self.G = g_model
        self.E = e_model
        self.D = d_model
        self.merge = merge
        self.levels = int(np.log2(self.x_dim/2))  # the number of blocks

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

        if self.merge:
            self.alpha = 0
            self.alpha_step = alpha_step
            self.G.get_layer("Fade_G").trainable = False
            self.E.get_layer("Fade_E").trainable = False
            self.E.get_layer("Fade_E_w").trainable = False

        # get trainable params
        self.θ_F = self.F.trainable_weights
        self.θ_G = self.G.trainable_weights
        self.θ_E = self.E.trainable_weights
        self.θ_D = self.D.trainable_weights

    def real_as_tensor(self, x_real):
        return tf.Variable(x_real, dtype=tf.float32)

    @tf.function
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
        if self.merge:
            self.alpha += self.alpha_step
            K.set_value(self.G.get_layer("Fade_G").alpha, self.alpha)
            K.set_value(self.E.get_layer("Fade_E").alpha, self.alpha)
            K.set_value(self.E.get_layer("Fade_E_w").alpha, self.alpha)

        # -------------------------------#
        # Step I - Update Discriminator  #
        # -------------------------------#

        # Random mini-batch from data set
        x_real, noise, constant = batch

        # samples from prior N(0, 1)
        z = K.random_normal((batch_size, self.z_dim))
        # generate fake images
        x_fake = self.generator([z, noise, constant])

        # Compute loss and apply gradients
        with tf.GradientTape() as tape:

            fake_pred = self.discriminator(x_fake)

            real_pred = self.discriminator(x_real)

            loss_d = discriminator_logistic_non_saturating(real_pred, fake_pred)

            # Add the R1 term
            if self.γ > 0:

                x_real = self.real_as_tensor(x_real) #tf.Variable(x_real, dtype=tf.float32)
                with tf.GradientTape() as r1_tape:
                    r1_tape.watch(x_real)
                    # 1. Get the discriminator output for real images
                    pred = self.discriminator(x_real)

                # 2. Calculate the gradients w.r.t to the real images.
                grads = r1_tape.gradient(pred, [x_real])[0]

                # 3. Calculate the squared norm of the gradients
                r1_penalty = K.sum(K.square(grads))
                loss_d += self.γ / 2 * r1_penalty

        gradients = tape.gradient(loss_d, self.θ_E + self.θ_D)
        self.optimizer_d.apply_gradients(zip(gradients, self.θ_E + self.θ_D))

        # ----------------------------#
        #  Step II - Update Generator #
        # ----------------------------#

        # samples from prior N(0, 1)
        z = K.random_normal((batch_size, self.z_dim))
        # Compute loss and apply gradients
        with tf.GradientTape() as tape:

            fake_pred = self.discriminator(self.generator([z, noise, constant]))

            loss_g = generator_logistic_non_saturating(fake_pred)

        gradients = tape.gradient(loss_g, self.θ_F + self.θ_G)
        self.optimizer_g.apply_gradients(zip(gradients, self.θ_F + self.θ_G))

        # ------------------------------#
        #  Step III - Update Reciprocal #
        # ------------------------------#

        # samples from prior N(0, 1)
        z = K.random_normal((batch_size, self.z_dim))
        # Get w
        w = self.F(z)
        # Compute loss and apply gradients
        with tf.GradientTape() as tape:

            w_pred = self.reciprocal([w, noise, constant])

            loss_r = l2(w, w_pred)

        gradients = tape.gradient(loss_r, self.θ_G + self.θ_E)
        self.optimizer_r.apply_gradients(zip(gradients, self.θ_G + self.θ_E))

        return {"loss_d": loss_d, "loss_g": loss_g, "loss_r": loss_r, "loss_gp": r1_penalty}

    def call(self, inputs):
        return inputs


# --------------------- DISTRIBUTED TRAINING ---------------------------------------------- #
# How to train using distributed https://www.tensorflow.org/tutorials/distribute/custom_training
# There appears to be issues with multi-gpu training. Instead of training inside model.fit method,
# this instead attempts to wrap each training step with @tf.function...
# -------------------------------#

def create_discriminator_training_step(model, batch_size):

    @tf.function
    def discriminator_train_step(batch):
        # Step I - Update Discriminator  #
        # -------------------------------#

        # Random mini-batch from data set
        x_real, noise, constant = batch

        # samples from prior N(0, 1)
        z = K.random_normal((batch_size, model.z_dim))
        # generate fake images
        x_fake = model.generator([z, noise, constant])

        # Compute loss and apply gradients
        with tf.GradientTape() as tape:

            fake_pred = model.discriminator(x_fake)

            real_pred = model.discriminator(x_real)

            loss_d = discriminator_logistic_non_saturating(real_pred, fake_pred)

            # Add the R1 term
            if model.γ > 0:

                x_real = model.real_as_tensor(x_real) #tf.Variable(x_real, dtype=tf.float32)
                with tf.GradientTape() as r1_tape:
                    r1_tape.watch(x_real)
                    # 1. Get the discriminator output for real images
                    pred = model.discriminator(x_real)

                # 2. Calculate the gradients w.r.t to the real images.
                grads = r1_tape.gradient(pred, [x_real])[0]

                # 3. Calculate the squared norm of the gradients
                r1_penalty = tf.nn.compute_average_loss(K.sum(K.square(grads), axis=[1, 2, 3]))
                loss_d += model.γ / 2 * r1_penalty

        gradients = tape.gradient(loss_d, model.θ_E + model.θ_D)
        model.optimizer_d.apply_gradients(zip(gradients, model.θ_E + model.θ_D))

        return loss_d

    return discriminator_train_step


def create_generator_training_step(model, batch_size):
    @tf.function
    def generator_train_step(batch):
        # ----------------------------#
        #  Step II - Update Generator #
        # ----------------------------#
        x_real, noise, constant = batch
        # samples from prior N(0, 1)
        z = K.random_normal((batch_size, model.z_dim))
        # Compute loss and apply gradients
        with tf.GradientTape() as tape:
            fake_pred = model.discriminator(model.generator([z, noise, constant]))

            loss_g = generator_logistic_non_saturating(fake_pred)

        gradients = tape.gradient(loss_g, model.θ_F + model.θ_G)
        model.optimizer_g.apply_gradients(zip(gradients, model.θ_F + model.θ_G))

        return loss_g

    return generator_train_step


def create_reciprocal_training_step(model, batch_size):
    @tf.function
    def reciprocal_train_step(batch):
        # ------------------------------#
        #  Step III - Update Reciprocal #
        # ------------------------------#
        x_real, noise, constant = batch
        # samples from prior N(0, 1)
        z = K.random_normal((batch_size, model.z_dim))
        # Get w
        w = model.F(z)
        # Compute loss and apply gradients
        with tf.GradientTape() as tape:
            w_pred = model.reciprocal([w, noise, constant])

            loss_r = l2(w, w_pred)

        gradients = tape.gradient(loss_r, model.θ_G + model.θ_E)
        model.optimizer_r.apply_gradients(zip(gradients, model.θ_G + model.θ_E))

        return loss_r

    return reciprocal_train_step


def create_distributed_train_step(train_step, strategy):
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
    return distributed_train_step


