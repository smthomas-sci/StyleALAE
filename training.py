
from StyleALAE.losses import *


# --------------------- DISTRIBUTED TRAINING ---------------------------------------------- #
# How to train using distributed https://www.tensorflow.org/tutorials/distribute/custom_training
# There appears to be issues with multi-gpu training. Instead of training inside model.fit method,
# this instead attempts to wrap each training step with @tf.function...
# https://www.tensorflow.org/guide/distributed_training#examples_and_tutorials
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
