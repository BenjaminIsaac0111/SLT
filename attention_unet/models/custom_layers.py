import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Wrapper, InputSpec


class GroupNormalization(layers.Layer):
    def __init__(self, groups=32, epsilon=1e-5, **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.gamma = None
        self.beta = None
        self.groups = groups
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True,
            name='gamma'
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True,
            name='beta'
        )

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, height, width, channels = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        group_shape = [batch_size, height, width, self.groups, channels // self.groups]

        inputs = tf.reshape(inputs, group_shape)
        mean, variance = tf.nn.moments(inputs, [1, 2, 4], keepdims=True)
        inputs = (inputs - mean) / tf.sqrt(variance + self.epsilon)

        inputs = tf.reshape(inputs, input_shape)
        return self.gamma * inputs + self.beta

    def get_config(self):
        config = super(GroupNormalization, self).get_config()
        config.update({
            "groups": self.groups,
            "epsilon": self.epsilon,
        })
        return config


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


class AttentionBlock(layers.Layer):
    def __init__(self, inter_channel, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.attention_multiply = None
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


class DropoutAttentionBlock(layers.Layer):
    def __init__(self, inter_channel, **kwargs):
        super(DropoutAttentionBlock, self).__init__(**kwargs)
        self.upsample = None
        self.psi_conv = None
        self.add_xg_leaky_relu = None
        self.add_xg = None
        self.theta_x_gn = None
        self.theta_x_conv = None
        self.phi_g_gn = None
        self.phi_g_conv = None
        self.attention_multiply = None
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

        self.psi_conv = SpatialConcreteDropout(layer=layers.Conv2D(
            filters=1,
            kernel_size=1,
            padding='same',
            activation='sigmoid'
        ))

        self.upsample = layers.UpSampling2D(2, interpolation='bilinear')
        self.attention_multiply = layers.Multiply()

        super(DropoutAttentionBlock, self).build(input_shape)

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
        config = super(DropoutAttentionBlock, self).get_config()
        config.update({
            'inter_channel': self.inter_channel
        })
        return config


class ConcreteDropout(Wrapper):
    """
    A custom Keras layer wrapper implementing Concrete Dropout, a form of dropout with a learnable dropout rate.

    Args:
        layer (Layer): The layer to apply Concrete Dropout to.
        weight_regularizer (float): The weight regularization coefficient.
        dropout_regularizer (float): The dropout regularization coefficient.
        init_min (tf.float): The minimum initial value for the dropout rate.
        init_max (tf.float): The maximum initial value for the dropout rate.
        is_mc_dropout (bool): Whether to use Monte Carlo dropout.
        data_format (str): The data format, either 'channels_last' or 'channels_first'.
        temperature (float): The temperature parameter for the Concrete Dropout.
        **kwargs: Additional keyword arguments for the Wrapper base class.
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, data_format='channels_last', temperature=0.1,
                 **kwargs):
        assert 'kernel_regularizer' not in kwargs, "Must not provide a kernel regularizer."
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.temperature = temperature
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.init_min = tf.math.log(tf.constant(init_min)) - tf.math.log(
            tf.constant(1. - init_min))
        self.init_max = tf.math.log(tf.constant(init_max)) - tf.math.log(
            tf.constant(1. - init_max))
        self.data_format = data_format

    def build(self, input_shape=None):
        """
        Builds the layer by initializing the learnable dropout probability.

        Args:
            input_shape (tuple): The shape of the input tensor.
        """
        self.input_spec = InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()

        # Initialize p (learnable dropout probability)
        self.p_logit = self.add_weight(name='p_logit',
                                       shape=(1,),
                                       initializer=tf.random_uniform_initializer(
                                           self.init_min, self.init_max),
                                       trainable=True)

    def _get_noise_shape(self, inputs):
        """
        Returns the shape of the noise to be added for dropout.

        This method must be implemented by subclasses.

        Args:
            inputs (Tensor): The input tensor.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError(
            "Subclasses of ConcreteDropout must implement the noise shape")

    def compute_output_shape(self, input_shape):
        """
        Computes the output shape of the layer.

        Args:
            input_shape (tuple): The shape of the input tensor.

        Returns:
            tuple: The shape of the output tensor.
        """
        return self.layer.compute_output_shape(input_shape)

    def spatial_concrete_dropout(self, x, p):
        """
        Applies Concrete Dropout to the input tensor.

        Args:
            x (Tensor): The input tensor.
            p (Tensor): The dropout probability.

        Returns:
            Tensor: The output tensor after applying dropout.
        """
        eps = tf.keras.backend.epsilon()
        noise_shape = self._get_noise_shape(x)
        unif_noise = tf.random.uniform(shape=noise_shape, dtype=tf.float16)
        drop_prob = (
                tf.math.log(p + eps)
                - tf.math.log1p(-p + eps)
                + tf.math.log(unif_noise + eps)
                - tf.math.log1p(-unif_noise + eps)
        )
        drop_prob = tf.sigmoid(drop_prob / self.temperature)
        random_tensor = 1. - drop_prob
        retain_prob = 1. - p
        x = x * random_tensor
        x = x / retain_prob

        return x

    def call(self, inputs, training=None):
        """
        Calls the layer on a given input.

        Args:
            inputs (Tensor): The input tensor.
            training (bool): Whether the layer should behave in training mode or inference mode.

        Returns:
            Tensor: The output tensor.
        """
        p = tf.sigmoid(self.p_logit)
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * tf.reduce_sum(tf.square(weight)) / (1. - p)
        dropout_regularizer = p * tf.math.log(p)
        dropout_regularizer += (1. - p) * tf.math.log1p(-p)
        dropout_regularizer *= self.dropout_regularizer * self.input_dim
        regularizer = tf.reduce_sum(kernel_regularizer + dropout_regularizer)

        self.layer.add_loss(regularizer)

        if self.is_mc_dropout:
            return self.layer.call(self.spatial_concrete_dropout(inputs, p))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.spatial_concrete_dropout(inputs, p))

            return tf.keras.backend.in_train_phase(relaxed_dropped_inputs,
                                                   self.layer.call(inputs),
                                                   training=training)

    def get_config(self):
        """
        Returns the config of the layer.

        Returns:
            dict: The configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'layer': tf.keras.layers.serialize(self.layer),
            'weight_regularizer': self.weight_regularizer,
            'dropout_regularizer': self.dropout_regularizer,
            'init_min': float(self.init_min),
            'init_max': float(self.init_max),
            'is_mc_dropout': self.is_mc_dropout,
            'data_format': self.data_format,
            'temperature': self.temperature
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a layer from its config.

        Args:
            config (dict): The config dictionary.

        Returns:
            ConcreteDropout: A ConcreteDropout instance.
        """
        layer = tf.keras.layers.deserialize(config.pop('layer'))
        return cls(layer, **config)


class SpatialConcreteDropout(ConcreteDropout):
    """
    A custom Keras layer wrapper implementing Spatial Concrete Dropout, a form of Concrete Dropout for Conv2D layers.

    Args:
        layer (Layer): The layer to apply Spatial Concrete Dropout to.
        temperature (float): The temperature parameter for the Concrete Dropout.
        **kwargs: Additional keyword arguments for the ConcreteDropout base class.
    """

    def __init__(self, layer, temperature=2. / 3., **kwargs):
        super(SpatialConcreteDropout, self).__init__(
            layer, temperature=temperature, **kwargs)

    def _get_noise_shape(self, inputs):
        """
        Returns the shape of the noise to be added for dropout.

        Args:
            inputs (Tensor): The input tensor.

        Returns:
            tuple: The shape of the noise tensor.
        """
        input_shape = tf.shape(inputs)
        if self.data_format == 'channels_first':
            return input_shape[0], input_shape[1], 1, 1
        elif self.data_format == 'channels_last':
            return input_shape[0], 1, 1, input_shape[-1]

    def build(self, input_shape=None):
        """
        Builds the layer by initializing the input specifications and validating the input shape.

        Args:
            input_shape (tuple): The shape of the input tensor.
        """
        self.input_spec = InputSpec(shape=input_shape)

        super(SpatialConcreteDropout, self).build(input_shape=input_shape)

        assert len(input_shape) == 4, 'this wrapper only supports Conv2D layers'
        if self.data_format == 'channels_first':
            self.input_dim = input_shape[1]  # we drop only channels
        else:
            self.input_dim = input_shape[3]

    def get_config(self):
        """
        Returns the config of the layer.

        Returns:
            dict: The configuration dictionary.
        """
        config = super().get_config()
        config.update({
            'temperature': self.temperature
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Creates a layer from its config.

        Args:
            config (dict): The config dictionary.

        Returns:
            SpatialConcreteDropout: A SpatialConcreteDropout instance.
        """
        layer = tf.keras.layers.deserialize(config.pop('layer'))
        return cls(layer, **config)
