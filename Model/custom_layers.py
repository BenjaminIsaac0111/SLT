import tensorflow as tf
from tensorflow.keras import layers
from tensorflow_addons.layers import GroupNormalization


class Linear(tf.keras.layers.Layer):
    def __init__(self, units=32, input_shape=None, name=None, **kwargs):
        """
        Linear layer implementation. NOTE: this is for compatibility with MIM/HGDL models. Pass into custom objects
        when loading the HGDL model.

        Args:
            units (int, optional): Dimensionality of the output space. Defaults to 32.
            input_shape (tuple, optional): Shape of the input tensor. Defaults to None.
            name (str, optional): Name of the layer. Defaults to None.
            **kwargs: Additional keyword arguments.

        Attributes:
            op_shape: Shape of the operation.
            units: Dimensionality of the output space.
            b: Bias variable.

        Methods:
            build: Builds the layer by creating the bias variable.
            get_config: Retrieves the layer configuration.
            call: Performs the forward pass of the layer.

        """
        super(Linear, self).__init__(kwargs, name=name)
        self.op_shape = input_shape
        self.units = units
        self.b = None

    def build(self, input_shape):
        """
        Builds the layer by creating the bias variable.

        Args:
            input_shape (tuple): Shape of the input tensor.

        """
        self.b = self.add_weight(
            name="b",
            shape=(self.units,),
            # initializer="random_normal",
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01),
            trainable=True
        )

    def get_config(self):
        """
        Retrieves the layer configuration.

        Returns:
            dict: Dictionary containing the layer configuration.

        """
        config = super(Linear, self).get_config()
        config['units'] = self.units
        config['input_shape'] = self.op_shape
        return dict(list(config.items()))

    def call(self, inputs):
        """
        Performs the forward pass of the layer.

        Args:
            inputs: Input tensor.

        Returns:
            tf.Tensor: Output tensor.

        """
        return tf.keras.layers.LeakyReLU(alpha=0.01)(
            inputs + tf.broadcast_to(self.b, [self.op_shape[0], self.op_shape[1], self.units]))


class MaxBlurPooling2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=2, kernel_size=3, **kwargs):
        super(MaxBlurPooling2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.gaussian_filter = self._create_gaussian_kernel(kernel_size)

    def _create_gaussian_kernel(self, size):
        """Creates a 2D Gaussian kernel."""
        d = tf.distributions.Normal(0., 1.)
        vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
        gauss_kernel = tf.einsum('i,j->ij', vals, vals)
        gauss_kernel = gauss_kernel / tf.reduce_sum(gauss_kernel)
        return gauss_kernel[:, :, tf.newaxis, tf.newaxis]

    def call(self, inputs):
        # Expand the Gaussian filter to match the input channels
        channels = tf.shape(inputs)[-1]
        gaussian_filter = tf.tile(self.gaussian_filter, [1, 1, channels, 1])

        # Apply Gaussian blur
        blur = tf.nn.depthwise_conv2d(inputs, gaussian_filter, strides=[1, 1, 1, 1], padding='SAME')

        # Apply max pooling
        output = tf.nn.max_pool(blur, ksize=[1, self.pool_size, self.pool_size, 1],
                                strides=[1, self.pool_size, self.pool_size, 1], padding='SAME')
        return output

    def get_config(self):
        config = super(MaxBlurPooling2D, self).get_config()
        config.update({
            'pool_size': self.pool_size,
            'kernel_size': self.kernel_size
        })
        return config


class PixelShuffle(layers.Layer):
    def __init__(self, upscale_factor, **kwargs):
        super(PixelShuffle, self).__init__(**kwargs)
        self.conv = None
        self.upscale_factor = upscale_factor

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=input_shape[-1] * (self.upscale_factor ** 2),
            kernel_size=3,
            padding='same',
            activation=tf.keras.layers.LeakyReLU()
        )

    def call(self, inputs):
        x = self.conv(inputs)
        x = tf.nn.depth_to_space(x, block_size=self.upscale_factor)
        return x

    def get_config(self):
        config = super(PixelShuffle, self).get_config()
        config.update({
            'upscale_factor': self.upscale_factor,
        })
        return config


class AttentionBlock(layers.Layer):
    def __init__(self, inter_channel, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.attention_multiply = None
        self.pixel_shuffle = None
        self.inter_channel = inter_channel

    def build(self, input_shape):
        self.phi_g_conv = layers.Conv2D(
            self.inter_channel,
            kernel_size=1,
            strides=1,
            padding='same'
        )
        self.phi_g_gn = GroupNormalization()

        self.theta_x_conv = layers.Conv2D(
            self.inter_channel,
            kernel_size=1,
            strides=2,
            padding='same'
        )
        self.theta_x_gn = GroupNormalization()

        self.add_xg = layers.Add()
        self.add_xg_leaky_relu = layers.LeakyReLU()

        self.psi_conv = layers.Conv2D(
            filters=1,
            kernel_size=1,
            padding='same',
            activation='sigmoid'
        )

        self.upsample = layers.UpSampling2D(2, interpolation='bilinear')
        self.attention_multiply = layers.Multiply()

        super(AttentionBlock, self).build(input_shape)

    def call(self, inputs):
        x, g = inputs
        phi_g = self.phi_g_conv(g)
        phi_g = self.phi_g_gn(phi_g)

        theta_x = self.theta_x_conv(x)
        theta_x = self.theta_x_gn(theta_x)

        add_xg = self.add_xg([theta_x, phi_g])
        add_xg = self.add_xg_leaky_relu(add_xg)

        psi = self.psi_conv(add_xg)
        psi = self.upsample(psi)
        output = self.attention_multiply([psi, x])

        return output

    def get_config(self):
        config = super(AttentionBlock, self).get_config()
        config.update({
            'inter_channel': self.inter_channel
        })
        return config
