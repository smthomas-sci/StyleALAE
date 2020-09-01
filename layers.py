"""
Custom layers to build an ALAE generative network.

Author: Simon Thomas
Date: Jul-6-2020


PYTHON SOFTWARE VERSIONS:
- tensorflow            2.2.0
- tensorflow-addons     0.10.0
- scikit-image          0.17.2
- numpy                 1.17.2
- matplotlib            3.1.3
- Keras                 2.3.1
- h5py                  2.10.0


CUDA
- cudatoolkit           10.1.243
- cudnn                 7.6.5
"""

import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np

from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Dense, LeakyReLU, Conv2D, Flatten, Add, Cropping2D, Layer
from tensorflow.keras.layers import UpSampling2D, Reshape, AveragePooling2D, Input
from tensorflow.keras.models import Model

# -------------------  FIXES BUG ---------------------------- #
# See https://github.com/tensorflow/tensorflow/issues/34983
# and comment from Papageno2
#tf.config.experimental_run_functions_eagerly(True)
# ---------------------------------------------------------- #

normal = tf.initializers.RandomNormal
ones = tf.initializers.ones


def crop_noise(noise_tensor, size, block):
    """
    Crops the noise_tensor to the target size.
    """
    cut = (noise_tensor.shape[1] - size) // 2
    crop = Cropping2D(cut, name=f"G_Noise_Crop_block_{block}")(noise_tensor)
    return crop


class DenseEQ(Dense):
    """
    Standard dense layer but includes learning rate equilization
    at runtime as per Karras et al. 2017. Includes learning rate multiplier,
    but defaults to 1.0. Only needed for the mapping network.

    Inherits Dense layer and overides the call method.
    """
    def __init__(self, lrmul=1.0, **kwargs):
        if 'kernel_initializer' in kwargs:
            raise Exception("Cannot override kernel_initializer")
        self.lrmul=lrmul
        super().__init__(kernel_initializer=normal(0, 1/self.lrmul), **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        # The number of inputs
        n = np.product([int(val) for val in input_shape[1:]])
        # He initialisation constant
        self.c = np.sqrt(2/n)*self.lrmul

    def call(self, inputs):
        output = K.dot(inputs, self.kernel*self.c) # scale kernel
        if self.use_bias:
            output = K.bias_add(output, self.bias*self.lrmul, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output


class Conv2DEQ(Conv2D):
    """
    Standard Conv2D layer but includes learning rate equilization
    at runtime as per Karras et al. 2017.

    Inherits Conv2D layer and overrides the call method, following
    https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py

    from the tf-keras branch.

    """

    def __init__(self, **kwargs):
        """
        Requires usual Conv2D inputs e.g.
         - filters, kernel_size, strides, padding
        """

        if 'kernel_initializer' in kwargs:
            raise Exception("Cannot override kernel_initializer")
        super().__init__(kernel_initializer=normal(0, 1), **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        # The number of inputs
        n = np.product([int(val) for val in input_shape[1:]])
        # He initialisation constant
        self.c = np.sqrt(2 / n)

    def call(self, inputs):
        outputs = K.conv2d(
            inputs,
            self.kernel * self.c,  # scale kernel
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class Fade(Add):
    """
    A Fade layer which performs a weighted sum of the
    TWO inputs. Set alpha through training with:
        K.set_value(layer.alpha, new_alpha)
    """
    # init with default value
    def __init__(self, alpha=0.0, **kwargs):
        super(Fade, self).__init__(**kwargs)
        self.alpha = tf.Variable(alpha,
                                 name='ws_alpha',
                                 trainable=False,
                                 dtype=tf.float32,
                                 aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA,
                                 )

    # output a weighted sum of inputs
    def _merge_function(self, inputs):
        # only supports a weighted sum of two inputs
        assert (len(inputs) == 2)
        # ((1-a) * input1) + (a * input2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output


class AdaInstanceNormalization(Layer):
    """
    This is the AdaInstanceNormalization layer used by manicman199 available at
     https://github.com/manicman1999/StyleGAN-Keras

     This is used in StyleGAN version 1 as well as ALAE.
    """

    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 **kwargs):
        super(AdaInstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center # always done
        self.scale = scale # always done

    def build(self, input_shape):

        dim = input_shape[0][self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape[0]) + '.')

        super(AdaInstanceNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs[0])
        reduction_axes = list(range(0, len(input_shape)))

        beta = inputs[1]
        gamma = inputs[2]

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]
        mean = K.mean(inputs[0], reduction_axes, keepdims=True)
        stddev = K.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs[0] - mean) / stddev

        return normed * gamma + beta

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale
        }
        base_config = super(AdaInstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Bias(Layer):
    """
    A simple bias layer used in StyleGAN2 after
    the Mod/Demod layer
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        # Conv bias
        self.bias = self.add_weight("bias",
                                    shape=[self.units, ],
                                    initializer=ones
                                    )

    def call(self, inputs, **kwargs ):
        return inputs + self.bias


class ModulationConv2D(Layer):
    """
    Modulation/Demodulation Convolutional layer, including learning rate equilization
    at runtime as per Karras et al. 2017 & 2019. (ProGAN & StyleGan2)

    Inspired by https://github.com/moono/stylegan2-tf-2.x/blob/master/stylegan2/custom_layers.py

    Look at tf-keras branch at
    https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py

    for implementation details.

    """

    def __init__(self, filters, kernel_size, style_fmaps, block, demodulate=False, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.style_fmaps = style_fmaps  # this is z_dim
        self.block = block
        self.demodulate = demodulate

    def build(self, input_shape):
        """
        Input shape is a list of input shapes.

        x_shape = input_shape[0] - the shape of the feature map
        w_shape = input_shape[1] - the shape of the style vector
        """
        x_shape, w_shape = input_shape[0], input_shape[1]

        # print("Build X-shape:", x_shape)

        # Conv Kernel
        self.kernel = self.add_weight("kernel",
                                      shape=(self.kernel_size[0], self.kernel_size[1],
                                             x_shape[-1], self.filters),
                                      initializer=normal(0, 1),
                                      trainable=True
                                      )

        # Equilized learning rate constant
        n = np.product([int(val) for val in x_shape[1:]])
        # He initialisation constant
        self.c = np.sqrt(2 / n)

        # add modulation layer
        self.modulate = DenseEQ(units=x_shape[-1], name=f"modulation_{self.block}")

    def scale_weights(self, style):
        """
        Scales and transforms the weights using the style vector

        B - BATCH
        k - kernel
        I - Input Features
        O - Output Features

        """
        # convolution kernel weights for fused conv
        weight = self.kernel * self.c  # [kkIO]
        weight = weight[np.newaxis]  # [BkkIO]

        # modulation
        style = self.modulate(style)  # [BI] - includes the bias
        weight *= style[:, np.newaxis, np.newaxis, :, np.newaxis]  # [BkkIO]

        # demodulation
        if self.demodulate:
            # demodulate with the L2 Norm of the weights (statistical assumption)
            d = tf.math.rsqrt(tf.reduce_sum(tf.square(weight), axis=[1, 2, 3]) + 1e-8)  # [BO]
            weight *= d[:, np.newaxis, np.newaxis, np.newaxis, :]  # [BkkIO]

        # weight: reshape, prepare for fused operation
        new_weight_shape = [tf.shape(weight)[1], tf.shape(weight)[2], tf.shape(weight)[3], -1]  # [kkI(BO)]
        weight = tf.transpose(weight, [1, 2, 3, 0, 4])  # [kkIBO]
        weight = tf.reshape(weight, shape=new_weight_shape)  # [kkI(BO)]
        return weight

    def call(self, inputs, **kwargs):
        x = inputs[0]
        style = inputs[1]

        # Transform the weights using the style vector
        weights = self.scale_weights(style)

        # Prepare inputs: reshape minibatch to convolution groups
        rows = x.shape[1]
        cols = x.shape[2]
        x = tf.reshape(x, [1, -1, rows, cols])

        # Perform convolution
        x = tf.nn.conv2d(x, weights, data_format='NCHW', strides=[1, 1, 1, 1], padding='SAME')

        # x: reshape back to batches
        x = tf.reshape(x, [-1, self.filters, tf.shape(x)[2], tf.shape(x)[3]])

        # x: reshape to [BHWO]
        x = tf.transpose(x, [0, 2, 3, 1])

        return x

    def get_config(self):
        config = super(ModulationConv2D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel': self.kernel,
            'style_fmaps': self.style_fmaps,
            'demodulate': self.demodulate,
            'up': self.up,
            'down': self.down
        })
        return config


class MeanAndStDev(Layer):
    """
    This is the Instance Normalization transformation which
    concatenates mu and sigma to later be mapped to w.

    This is used in the encoder introduced by
    """
    def __init__(self, **kwargs):
        super(MeanAndStDev, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MeanAndStDev, self).build(input_shape)

    def call(self, inputs, **kwargs):
        m = K.mean(inputs, axis=[1, 2], keepdims=True)
        std = K.std(inputs, axis=[1, 2], keepdims=True)
        statistics = K.concatenate([m, std], axis=1)
        return statistics



class EncoderBlock(Model):
    """
    Encoder block using instance normalisation to extract style
    as introduced in the ALAE by Podgorskiy et al (2020).
    """
    def __init__(self, filters, block, z_dim, **kwargs):
        """
        :param filters: the number of convolution filters (fixed 3x3 size)
        :param block: the block number for naming
        :param z_dim: the z-dimension for mapping features to style vector
        """
        super(EncoderBlock, self).__init__(**kwargs)
        # Attributes
        self.filters = filters
        self.block = block
        self.z_dim = z_dim

        # Trainable Layers
        self.conv1 = Conv2DEQ(filters=filters, kernel_size=(3, 3), padding="same", name=f"E_block_{block}_Conv_1")
        self.act1 = LeakyReLU(0.2, name=f"E_block_{block}_Act_1")
        self.msd = MeanAndStDev(name=f"E_block_{block}_msd")
        self.in1 = InstanceNormalization(name=f"E_block_{block}_IN_1", center=False, scale=False)
        self.in2 = InstanceNormalization(name=f"E_block_{block}_IN_2", center=False, scale=False)
        self.conv2 = Conv2DEQ(filters=filters, kernel_size=(3, 3), padding="same", name=f"E_block_{block}_Conv_2")
        self.act2 = LeakyReLU(0.2, name=f"E_block_{block}_Act_2")
        self.downsample = AveragePooling2D(name=f"E_block_{block}_DownSample")
        self.mapStyle1 = DenseEQ(units=z_dim, name=f"E_block_{block}_style_1")
        self.mapStyle2 = DenseEQ(units=z_dim, name=f"E_block_{block}_style_2")
        self.flatten = Flatten(name=f"E_block_{block}_flatten")

    def call(self, inputs, **kwargs):
        # Convolution 1
        x = self.conv1(inputs)
        x = self.act1(x)

        # Instance Normalisation 1
        style1 = self.flatten(self.msd(x))
        x = self.in1(x)

        # Convolution 2
        x = self.conv2(x)
        if self.block > 1:
            x = self.downsample(x)
        x = self.act2(x)

        # Instance Normalisation 2
        style2 = self.flatten(self.msd(x))
        x = self.in2(x)

        # Affine transform to style vectors
        w1 = self.mapStyle1(style1)
        w2 = self.mapStyle2(style2)

        return x, w1, w2


class EncoderBlockRouter(Model):
    """
    This is a wrapper that makes progressive growing possible
    with the complex input/output routing of the encoder block.
    """
    def __init__(self, filters, block, z_dim, **kwargs):
        """
        :param filters: a list of filters for the block - [input_filters, block_filters] e.g. [16, 32]
        :param block: the block number for naming
        :param z_dim: the z-dimension for mapping to a style vector
        """
        super(EncoderBlockRouter, self).__init__(**kwargs)
        # Attributes
        self.filters = filters
        self.block = block
        self.z_dim = z_dim

        # Parameters
        self.conv1 = Conv2DEQ(filters=self.filters[0], kernel_size=(3, 3), padding="same")
        self.act1 = LeakyReLU(alpha=0.2)
        self.encode = EncoderBlock(self.filters[1], self.block, self.z_dim, name=f"E_block_{block}_encoder")

    def call(self, inputs, **kwargs):
        x = inputs

        x = self.conv1(x)
        x = self.act1(x)

        x, w1, w2 = self.encode(x)

        return x, w1, w2


class GeneratorBlockModDemod(Model):
    """
    Generator block using Modulation and Demodulation of the
    weights to inject style (introduced in StyleGAN version 2).
    """
    def __init__(self, filters, block, z_dim, **kwargs):
        super(GeneratorBlockModDemod, self).__init__(**kwargs)
        # Attributes
        self.filters = filters
        self.block = block
        self.z_dim = z_dim
        self.dim = 2 ** (block + 1)

        # Parameters
        self.upsample = UpSampling2D(name=f"G_block_{block}_UpSample")
        self.noise = Conv2DEQ(filters=filters, kernel_size=1, padding='same', name=f"G_block_{block}_noise_bias")
        # phase 1
        self.conv1 = ModulationConv2D(filters, kernel_size=(3, 3), style_fmaps=self.z_dim,
                                      block=self.block, demodulate=False, name=f"G_block_{block}_ModulationConv_1")
        self.bias1 = Bias(units=self.filters, name=f"G_block_{block}_bias_1")
        self.addNoise1 = Add(name=f"G_block_{block}_Add_1")
        self.act1 = LeakyReLU(0.2, name=f"G_block_{block}_Act_1")

        if self.block > 1:  # > 4x4
            # phase 2
            self.conv2 = ModulationConv2D(filters, kernel_size=(3, 3), style_fmaps=self.z_dim,
                                          block=self.block, demodulate=False, name=f"G_block_{block}_ModulationConv_1")
            self.bias2 = Bias(units=self.filters, name=f"G_block_{block}_bias_2")
            self.addNoise2 = Add(name=f"G_block_{block}_Add_2")
            self.act2 = LeakyReLU(0.2, name=f"G_block_{block}_Act_2")

    def call(self, inputs, **kwargs):
        # Unpack inputs
        input_tensor, noise_tensor, style_tensor = inputs

        # Get noise image for level
        noise_tensor = crop_noise(noise_tensor, self.dim, self.block)

        if self.block > 1:
            x = self.upsample(input_tensor)
        else:
            x = input_tensor

        # Phase 1
        noise = self.noise(noise_tensor)
        x = self.conv1([x, style_tensor])
        x = self.bias1(x)
        x = self.addNoise1([x, noise])
        x = self.act1(x)

        if self.block == 1:  # 4x4 block
            return x

        # Phase 2
        x = self.conv2([x, style_tensor])
        x = self.bias2(x)
        x = self.addNoise2([x, noise])
        x = self.act2(x)

        return x


class GeneratorBlockAdaIN(Model):
    """
    Generator block using adaptive instance normalisation to inject style,
    as per StyleGAN version 1
    """
    def __init__(self, filters, block, z_dim=None, **kwargs):
        super(GeneratorBlockAdaIN, self).__init__(**kwargs)
        # Attributes
        self.filters = filters
        self.block = block
        self.dim = 2 ** (block + 1)
        self.z_dim = z_dim  # not necessary for this layer - helps with ModDemo block

        # Trainable Layers
        self.upsample = UpSampling2D(name=f"G_block_{block}_UpSample", interpolation="bilinear")
        # phase 1
        self.beta1 = DenseEQ(units=filters, name=f"G_block_{block}_beta1")
        self.beta1r = Reshape([1, 1, filters], name=f"G_block_{block}_beta1_reshape")
        self.gamma1 = DenseEQ(units=filters, name=f"G_block_{block}_gamma1")
        self.gamma1r = Reshape([1, 1, filters], name=f"G_block_{block}_gamma1_reshape")
        self.noise1 = Conv2DEQ(filters=filters, kernel_size=1, padding='same', name=f"G_block_{block}_noise_bias1")
        self.conv1 = Conv2DEQ(filters=filters, kernel_size=3, padding='same', name=f"G_block_{block}_decoder_conv1")
        self.AdaIn1 = AdaInstanceNormalization(name=f"G_block_{block}_AdaIN_1")
        self.addNoise1 = Add(name=f"G_block_{block}_Add_1")
        self.act1 = LeakyReLU(0.2, name=f"G_block_{block}_Act_1")

        # phase 2
        self.beta2 = DenseEQ(units=filters, name=f"G_block_{block}_beta2")
        self.beta2r = Reshape([1, 1, filters], name=f"G_block_{block}_beta2_reshape")
        self.gamma2 = Dense(units=filters, name=f"G_block_{block}_gamma2")
        self.gamma2r = Reshape([1, 1, filters], name=f"G_block_{block}_gamma2_reshape")
        self.noise2 = Conv2DEQ(filters=filters, kernel_size=1, padding='same', name=f"G_block_{block}_noise_bias2")
        self.conv2 = Conv2DEQ(filters=filters, kernel_size=3, padding='same', name=f"G_block_{block}_decoder_conv2")
        self.AdaIn2 = AdaInstanceNormalization(name=f"G_block_{block}_AdaIN_2")
        self.addNoise2 = Add(name=f"G_block_{block}_Add_2")
        self.act2 = LeakyReLU(0.2, name=f"G_block_{block}_Act_2")

    def call(self, inputs, **kwargs):
        # Unpack inputs
        input_tensor, noise_tensor, style_tensor = inputs

        # Get noise image for level
        noise_tensor = crop_noise(noise_tensor, self.dim, self.block)

        if self.block > 1:
            x = self.upsample(input_tensor)
        else:
            x = input_tensor

        # Phase 1
        beta = self.beta1r(self.beta1(style_tensor))
        gamma = self.gamma1r(self.gamma1(style_tensor))
        noise = self.noise1(noise_tensor)
        x = self.conv1(x)
        x = self.AdaIn1([x, beta, gamma])
        x = self.addNoise1([x, noise])
        x = self.act1(x)

        # Phase 2
        beta = self.beta2r(self.beta2(style_tensor))
        gamma = self.gamma2r(self.gamma2(style_tensor))
        noise = self.noise2(noise_tensor)
        x = self.conv2(x)
        x = self.AdaIn2([x, beta, gamma])
        x = self.addNoise2([x, noise])
        x = self.act2(x)

        return x


class tRGB(Layer):
    """
    Linear transformation from feature space to rgb space, using a 1x1 convolution
    """
    def __init__(self, block):
        super(tRGB, self).__init__()
        self.block = block
        self.transform = Conv2DEQ(filters=3, kernel_size=(1, 1), padding="same", name=f"G_block_{block}_tRGB")

    def build(self, input_shape):
        super(tRGB, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return self.transform(inputs)


# Progressive Growing Functions
def expand_encoder(old_model, filters, block, z_dim):
    """
    Expands the old model by a factor of 2. Size is not explicit and
    is determined by doubling the size of the input of the old_model

    :param old_model: the old "straight through" encoder model
    :param filters: a list of filters required EncoderBlockRouter - [input_filers, block_filters]
    :param block: the block number for naming
    :param z_dim: the z-dimension for mapping features to style vectors
    :return: straight_e (E_s), merged_e (E_m)
    """
    # Create new input
    dim = int(old_model.get_input_at(0).shape[1] * 2)
    c = old_model.get_input_at(0).shape[-1]
    e_in = Input(shape=(dim, dim, c), name="E_new_input")

    x_new, new_w1, new_w2 = EncoderBlockRouter(filters=filters, block=block,
                                               z_dim=z_dim, name=f"encoder_block_router_{block}")(e_in)

    # Save values
    x = x_new
    ws = [new_w1, new_w2]

    # Pass through remaining layers of old model
    for b in range(block - 1, 0, -1):
        if b == block - 1 and b != 0:
            x, w1, w2 = old_model.get_layer(f"encoder_block_router_{b}").encode(x)
        else:
            x, w1, w2 = old_model.get_layer(f"E_block_{b}_encoder")(x)

        ws.extend([w1, w2])

    e_out = Add(name="E_Final_Sum_Straight")(ws)

    straight_e = Model(inputs=[e_in], outputs=[e_out], name=f"E_straight_{dim}")

    # CREATE MERGE MODEL

    # Downsample input
    e_old_in = AveragePooling2D(name="downsample")(e_in)

    x = old_model.layers[1].layers[0](e_old_in)
    x_old = old_model.layers[1].layers[1](x)

    # Merge x into old model
    x = Fade(name="Fade_E")([x_old, x_new])

    # Pass through remaining layers of old model
    w_old = []
    for b in range(block - 1, 0, -1):
        if b == block - 1 and b != 0:
            x, w1, w2 = old_model.get_layer(f"encoder_block_router_{b}").encode(x)
        else:
            x, w1, w2 = old_model.get_layer(f"E_block_{b}_encoder")(x)

        w_old.extend([w1, w2])

    # Merge new W into old model
    w_old = Add(name="E_Sum_Old_W")(w_old)
    w_new = Add(name="E_Sum_New_W")([new_w1, new_w2, w_old])
    e_out = Fade(name="Fade_E_w")([w_old, w_new])

    merged_e = Model(inputs=[e_in], outputs=[e_out], name="E_merged")

    return straight_e, merged_e


def expand_generator(old_model, block, filters, z_dim, noise_dim, block_type="AdaIN"):
    """
    Expands the old model by increasing the output by a factor of 2. Size is not explicit and
    is determined by doubling the last feature map size.

    :param old_model: the old "straight through" generator model to expand
    :param block: the block number for naming
    :param filters: the number of convolution filters (all 3x3 size)
    :param z_dim: the number of z-dimensions to feed style vectors
    :param block_type: the type of generator block to use. Default is "AdaIN" else "ModDeMod
    :return: straight_g, merged_g
    """
    # Pre conditions
    assert (block > 1)
    GeneratorBlock = GeneratorBlockAdaIN if block_type == "AdaIN" else GeneratorBlockModDemod

    # Create new inputs
    w_inputs = [Input(shape=(z_dim,), name=f"G_w_input_{i + 1}") for i in range(block)]
    noise_input = Input(shape=(noise_dim, noise_dim, 1), name="G_noise_input")
    constant_input = Input(shape=(1, 1), name="G_constant_input")

    # Pass through old model up to tRGB
    noise = noise_input
    constant = constant_input
    x = old_model.get_layer(f"G_base")(constant)
    x = old_model.get_layer("G_base_reshape")(x)
    for b in range(block - 1):
        style = w_inputs[b]
        x = old_model.get_layer(f"G_block_{b + 1}_style")([x, noise, style])

    # Get old RGB and upsample
    old_out = old_model(w_inputs[:block - 1] + [noise_input, constant])
    old_out = UpSampling2D()(old_out)

    # Add new block
    style = w_inputs[block - 1]
    x = GeneratorBlock(filters=filters, block=block, z_dim=z_dim, name=f"G_block_{block}_style")([x, noise, style])

    # Transform to RGB
    new_out = tRGB(block)(x)

    # STRAIGHT MODEL
    g_inputs = w_inputs + [noise_input, constant_input]
    straight_g = Model(inputs=g_inputs, outputs=[new_out], name=f"G_straight_{block}")

    # MERGE MODEL
    g_out = Fade(name="Fade_G")([old_out, new_out])

    merged_g = Model(inputs=g_inputs, outputs=[g_out], name=f"G_merged_{block}")

    return straight_g, merged_g


