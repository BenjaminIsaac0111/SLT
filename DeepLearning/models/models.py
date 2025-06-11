from typing import Optional, Callable, Union, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import Regularizer


# =============================================================================
# Custom Layers
# =============================================================================

class GroupNormalization(layers.Layer):
    def __init__(self, groups: int = 32, epsilon: float = 1e-5, **kwargs):
        """
        Group Normalization layer.

        Args:
            groups: Number of groups to divide the channels.
            epsilon: Small constant to avoid division by zero.
            **kwargs: Additional keyword arguments.
        """
        super(GroupNormalization, self).__init__(**kwargs)
        self.groups = groups
        self.epsilon = epsilon

    def build(self, input_shape: tf.TensorShape):
        channels = int(input_shape[-1])
        # If the number of channels is less than the default groups,
        # use the number of channels as the number of groups.
        if channels < self.groups:
            self.groups = channels
        if channels % self.groups != 0:
            raise ValueError("Number of channels must be divisible by the number of groups. "
                             f"Got {channels} channels and {self.groups} groups.")
        self.gamma = self.add_weight(
            shape=(channels,),
            initializer='ones',
            trainable=True,
            name='gamma'
        )
        self.beta = self.add_weight(
            shape=(channels,),
            initializer='zeros',
            trainable=True,
            name='beta'
        )
        super(GroupNormalization, self).build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        input_shape = tf.shape(inputs)
        batch_size, height, width, channels = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        group_channels = channels // self.groups
        group_shape = [batch_size, height, width, self.groups, group_channels]
        x = tf.reshape(inputs, group_shape)
        mean, variance = tf.nn.moments(x, [1, 2, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(variance + self.epsilon)
        x = tf.reshape(x, tf.shape(inputs))
        return self.gamma * x + self.beta

    def get_config(self) -> dict:
        config = super(GroupNormalization, self).get_config()
        config.update({"groups": self.groups, "epsilon": self.epsilon})
        return config


# --- Attention Blocks ---

class BaseAttentionBlock(layers.Layer):
    """
    Base class for attention blocks.
    """

    def __init__(self, inter_channel: int, downsample: bool = True, **kwargs):
        """
        Args:
            inter_channel: Number of intermediate channels.
            downsample: Whether to downsample the first input (skip connection) via strides.
            **kwargs: Additional keyword arguments.
        """
        super(BaseAttentionBlock, self).__init__(**kwargs)
        self.inter_channel = inter_channel
        self.downsample = downsample

    def get_config(self) -> dict:
        config = super(BaseAttentionBlock, self).get_config()
        config.update({'inter_channel': self.inter_channel, 'downsample': self.downsample})
        return config


class AttentionBlock(BaseAttentionBlock):
    """
    Standard attention block.
    """

    def build(self, input_shape):
        stride = 2 if self.downsample else 1
        self.phi_g_conv = layers.Conv2D(self.inter_channel, kernel_size=1, strides=1, padding='same')
        self.phi_g_gn = GroupNormalization()
        self.theta_x_conv = layers.Conv2D(self.inter_channel, kernel_size=1, strides=stride, padding='same')
        self.theta_x_gn = GroupNormalization()
        self.add_xg = layers.Add()
        self.add_xg_leaky_relu = layers.LeakyReLU()
        self.psi_conv = layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')
        # When downsample is True, we upsample psi to match x; otherwise, no upsampling is needed.
        self.upsample = layers.UpSampling2D(2, interpolation='bilinear') if self.downsample else lambda z: z
        self.attention_multiply = layers.Multiply()
        super(AttentionBlock, self).build(input_shape)

    def call(self, inputs: list) -> tf.Tensor:
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


class DropoutAttentionBlock(BaseAttentionBlock):
    """
    Attention block with Spatial Concrete Dropout.
    """

    def build(self, input_shape):
        stride = 2 if self.downsample else 1
        self.phi_g_conv = layers.Conv2D(self.inter_channel, kernel_size=1, strides=1, padding='same')
        self.phi_g_gn = GroupNormalization()
        self.theta_x_conv = layers.Conv2D(self.inter_channel, kernel_size=1, strides=stride, padding='same')
        self.theta_x_gn = GroupNormalization()
        self.add_xg = layers.Add()
        self.add_xg_leaky_relu = layers.LeakyReLU()
        self.psi_conv = SpatialConcreteDropout(
            layer=layers.Conv2D(filters=1, kernel_size=1, padding='same', activation='sigmoid')
        )
        self.upsample = layers.UpSampling2D(2, interpolation='bilinear') if self.downsample else lambda z: z
        self.attention_multiply = layers.Multiply()
        super(DropoutAttentionBlock, self).build(input_shape)

    def call(self, inputs: list) -> tf.Tensor:
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


# --- Concrete Dropout Layers ---

class ConcreteDropout(layers.Wrapper):
    """
    A Keras layer wrapper that implements Concrete Dropout with a learnable dropout rate.
    """

    def __init__(
            self,
            layer: layers.Layer,
            weight_regularizer: float = 1e-6,
            dropout_regularizer: float = 1e-5,
            init_min: float = 0.1,
            init_max: float = 0.1,
            is_mc_dropout: bool = True,
            data_format: str = 'channels_last',
            temperature: float = 0.1,
            **kwargs
    ):
        if 'kernel_regularizer' in kwargs:
            raise ValueError("Do not provide a kernel_regularizer; it is computed internally.")
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.temperature = temperature
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.init_min = tf.math.log(tf.constant(init_min, dtype=tf.float32)) - tf.math.log(
            tf.constant(1. - init_min, dtype=tf.float32))
        self.init_max = tf.math.log(tf.constant(init_max, dtype=tf.float32)) - tf.math.log(
            tf.constant(1. - init_max, dtype=tf.float32))
        self.data_format = data_format
        self.input_dim = None

    def build(self, input_shape: tf.TensorShape = None):
        self.input_spec = layers.InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()
        self.p_logit = self.add_weight(
            name='p_logit',
            shape=(1,),
            initializer=tf.random_uniform_initializer(minval=self.init_min, maxval=self.init_max),
            trainable=True,
            dtype=tf.float32  # Force p_logit to remain in fp32
        )
        if self.data_format == 'channels_last':
            self.input_dim = input_shape[-1]
        elif self.data_format == 'channels_first':
            self.input_dim = input_shape[1]
        else:
            raise ValueError(f"Unsupported data_format: {self.data_format}")

    def _get_noise_shape(self, inputs: tf.Tensor) -> Tuple:
        raise NotImplementedError("Subclasses must implement _get_noise_shape.")

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        return self.layer.compute_output_shape(input_shape)

    def _compute_regularization(self, weight: tf.Tensor, p: tf.Tensor) -> tf.Tensor:
        """
        Computes the combined regularization loss.
        """
        # Weight regularizer: sum of squares of weights scaled by the dropout rate.
        weight_reg = self.weight_regularizer * tf.reduce_sum(tf.square(weight)) / (1. - p)
        # Dropout regularizer: encourages the learned dropout probability p
        dropout_reg = p * tf.math.log(p + 1e-10) + (1. - p) * tf.math.log(1. - p + 1e-10)
        dropout_reg *= self.dropout_regularizer * self.input_dim
        # Squeeze to ensure we return a scalar.
        return tf.squeeze(weight_reg + dropout_reg)

    def spatial_concrete_dropout(self, x: tf.Tensor, p: tf.Tensor) -> tf.Tensor:
        # Ensure that p is in float32 even if x is float16.
        p = tf.cast(p, tf.float32)

        # Define constants explicitly in float32.
        eps = tf.constant(tf.keras.backend.epsilon(), dtype=tf.float32)
        one = tf.constant(1.0, dtype=tf.float32)

        # Get the noise shape and generate uniform noise in float32.
        noise_shape = self._get_noise_shape(x)
        unif_noise = tf.random.uniform(shape=noise_shape, dtype=tf.float32)

        # Compute each term individually.
        log_p = tf.math.log(p + eps)  # log(p + eps)
        log_one_minus_p = tf.math.log(one - p + eps)  # log(1 - p + eps)
        log_unif_noise = tf.math.log(unif_noise + eps)  # log(unif_noise + eps)
        log_one_minus_unif_noise = tf.math.log(one - unif_noise + eps)  # log(1 - unif_noise + eps)

        # Combine the terms to compute drop_logit.
        drop_logit = log_p - log_one_minus_p + log_unif_noise - log_one_minus_unif_noise

        # Compute dropout probability.
        drop_prob = tf.sigmoid(drop_logit / self.temperature)
        random_tensor = one - drop_prob

        # Compute retain probability.
        retain_prob = one - p

        # Cast the dropout mask and retain probability to the input's dtype (e.g., float16).
        random_tensor = tf.cast(random_tensor, x.dtype)
        retain_prob = tf.cast(retain_prob, x.dtype)

        # Apply dropout scaling.
        return x * random_tensor / retain_prob

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        p = tf.sigmoid(self.p_logit)
        weight = self.layer.kernel
        reg_loss = self._compute_regularization(weight, p)
        self.layer.add_loss(reg_loss)

        def dropped_inputs():
            return self.layer.call(self.spatial_concrete_dropout(inputs, p))

        if self.is_mc_dropout:
            return self.layer.call(self.spatial_concrete_dropout(inputs, p))
        else:
            return tf.keras.backend.in_train_phase(dropped_inputs, self.layer.call(inputs), training=training)

    def get_config(self) -> dict:
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
    def from_config(cls, config: dict):
        layer = tf.keras.layers.deserialize(config.pop('layer'))
        return cls(layer, **config)


class SpatialConcreteDropout(ConcreteDropout):
    """
    A Concrete Dropout wrapper specialized for spatial dropout on Conv2D layers.
    """

    def __init__(self, layer: layers.Layer, temperature: float = 2. / 3., **kwargs):
        super(SpatialConcreteDropout, self).__init__(layer, temperature=temperature, **kwargs)

    def _get_noise_shape(self, inputs: tf.Tensor) -> Tuple:
        input_shape = tf.shape(inputs)
        if self.data_format == 'channels_first':
            return (input_shape[0], input_shape[1], 1, 1)
        elif self.data_format == 'channels_last':
            return (input_shape[0], 1, 1, input_shape[-1])
        else:
            raise ValueError(f"Unsupported data_format: {self.data_format}")

    def build(self, input_shape: tf.TensorShape = None):
        self.input_spec = layers.InputSpec(shape=input_shape)
        super(SpatialConcreteDropout, self).build(input_shape=input_shape)
        if len(input_shape) != 4:
            raise ValueError("SpatialConcreteDropout only supports 4D inputs (e.g., from Conv2D layers).")
        if self.data_format == 'channels_first':
            self.input_dim = input_shape[1]
        else:
            self.input_dim = input_shape[3]

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({'temperature': self.temperature})
        return config

    @classmethod
    def from_config(cls, config: dict):
        layer = tf.keras.layers.deserialize(config.pop('layer'))
        return cls(layer, **config)


# =============================================================================
# U-Net Builder Classes
# =============================================================================

class UnetBuilder:
    def __init__(
            self,
            input_size: tuple = (1024, 512, 3),
            num_classes: int = 9,
            num_levels: int = 4,
            num_conv_per_level: int = 3,
            num_filters: int = 64,
            regularisation: Optional[Regularizer] = None,
            use_attention: bool = True,
            activation: Union[str, Callable] = 'tanh',
            return_logits: bool = True,
            dropout_wrapper: Optional[Callable] = None,
    ):
        self.input_size = input_size
        self.num_classes = num_classes
        self.num_levels = num_levels
        self.num_conv_per_level = num_conv_per_level
        self.num_filters = num_filters
        self.regularisation = regularisation
        self.use_attention = use_attention
        self.activation = activation
        self.return_logits = return_logits
        self.dropout_wrapper = dropout_wrapper

    def conv_gn_block(self, x: tf.Tensor, filters: int, kernel_size: int, name_prefix: str) -> tf.Tensor:
        if self.dropout_wrapper:
            x = self.dropout_wrapper(
                name=f"{name_prefix}_dropout",
                layer=layers.Conv2D(
                    filters,
                    kernel_size,
                    activation=self.activation,
                    padding="same",
                    kernel_regularizer=self.regularisation,
                )
            )(x)
        else:
            x = layers.Conv2D(
                filters,
                kernel_size,
                activation=self.activation,
                padding="same",
                kernel_regularizer=self.regularisation,
                name=name_prefix
            )(x)
        x = GroupNormalization(name=f"{name_prefix}_gn")(x)
        return x

    def encoder_block(self, x: tf.Tensor, prev_output: tf.Tensor, level: int) -> Tuple[tf.Tensor, tf.Tensor]:
        for conv in range(self.num_conv_per_level):
            x = self.conv_gn_block(x, self.num_filters, 5, f'encoder_level_{level}_conv_{conv}')
        skip = x
        x = layers.AvgPool2D(2, name=f'encoder_level_{level}_pooling')(x)
        if self.dropout_wrapper:
            residual = self.dropout_wrapper(
                name=f'encoder_residual_level_{level}',
                layer=layers.Conv2D(
                    self.num_filters, 1, strides=2, padding='same', kernel_regularizer=self.regularisation
                )
            )(prev_output)
        else:
            residual = layers.Conv2D(
                self.num_filters, 1, strides=2, padding='same', kernel_regularizer=self.regularisation,
                name=f'encoder_residual_connection_{level}'
            )(prev_output)
        x = GroupNormalization(name=f'encoder_residual_gn_{level}')(x)
        x = layers.Add(name=f'residual_add_{level}')([x, residual])
        return x, skip

    def middle_block(self, x: tf.Tensor) -> tf.Tensor:
        for i in range(2):
            x = self.conv_gn_block(x, self.num_filters, 5, f'middle_conv_{i}')
        return x

    def apply_attention_block(self, skip: tf.Tensor, gating: tf.Tensor, level: int) -> tf.Tensor:
        if self.use_attention:
            if self.dropout_wrapper:
                return DropoutAttentionBlock(inter_channel=self.num_filters // 2, downsample=False,
                                             name=f'attention_block_{level}')([skip, gating])
            else:
                return AttentionBlock(inter_channel=self.num_filters // 2, downsample=False,
                                      name=f'attention_block_{level}')([skip, gating])
        return skip

    def decoder_block(self, x: tf.Tensor, skip: tf.Tensor, level: int) -> tf.Tensor:
        x = layers.UpSampling2D(2, name=f'decoder_upsampling_{level}', interpolation='bilinear')(x)
        skip_att = self.apply_attention_block(skip, x, level)
        x = layers.Concatenate(name=f'decoder_concat_{level}')([x, skip_att])
        if self.dropout_wrapper:
            residual = self.dropout_wrapper(
                name=f'decoder_residual_level_{level}',
                layer=layers.Conv2D(
                    self.num_filters, 1, activation=self.activation, padding='same',
                    kernel_regularizer=self.regularisation
                )
            )(x)
        else:
            residual = layers.Conv2D(
                self.num_filters, 1, activation=self.activation, padding='same', kernel_regularizer=self.regularisation,
                name=f'decoder_residual_connection_{level}'
            )(x)
        x = GroupNormalization(name=f'decoder_residual_gn_{level}')(x)
        for conv in range(self.num_conv_per_level):
            x = self.conv_gn_block(x, self.num_filters, 5, f'decoder_level_{level}_conv_{conv}')
        x = layers.Add(name=f'decoder_residual_add_{level}')([x, residual])
        return x

    def build_model(self) -> keras.Model:
        inputs = keras.Input(shape=self.input_size)
        x = inputs

        # Initial Convolution
        x = layers.Conv2D(
            self.num_filters,
            5,
            activation=self.activation,
            padding='same',
            kernel_regularizer=self.regularisation,
            name='initial_conv'
        )(x)
        x = GroupNormalization(name='initial_gn')(x)
        prev_output = x

        # Encoder
        skip_connections = []
        for level in range(self.num_levels):
            x, skip = self.encoder_block(x, prev_output, level)
            skip_connections.append(skip)
            prev_output = x

        # Middle Block
        x = self.middle_block(x)

        # Decoder
        for level in reversed(range(self.num_levels)):
            skip = skip_connections.pop()
            x = self.decoder_block(x, skip, level)

        # Output Convolution
        logits = layers.Conv2D(
            self.num_classes,
            1,
            activation=self.activation,
            padding="same",
            kernel_regularizer=self.regularisation,
            name='output_conv'
        )(x)
        outputs = layers.Softmax(dtype="float32", name='softmax_output')(logits)

        if self.return_logits:
            logits = tf.cast(logits, dtype=tf.float32)
            return keras.Model(inputs=inputs, outputs=[outputs, logits])
        else:
            return keras.Model(inputs=inputs, outputs=outputs)


class StandardUnetBuilder(UnetBuilder):
    def __init__(self, **kwargs):
        kwargs['dropout_wrapper'] = None
        super().__init__(**kwargs)


class MCDropoutUnetBuilder(UnetBuilder):
    def __init__(self, **kwargs):
        kwargs['dropout_wrapper'] = SpatialConcreteDropout
        super().__init__(**kwargs)


class EnsembleMCDropoutUnetBuilder(keras.Model):
    """
    A single Keras Model that creates multiple MCDropoutUnetBuilder sub-models and averages their outputs.
    """

    def __init__(
            self,
            n_models: int = 3,
            input_size: tuple = (1024, 512, 3),
            num_classes: int = 9,
            num_levels: int = 4,
            num_conv_per_level: int = 3,
            num_filters: int = 64,
            regularisation: Optional[keras.regularizers.Regularizer] = keras.regularizers.l2(),
            use_attention: bool = True,
            activation: Union[str, Callable] = 'tanh',
            return_logits: bool = True,
            name: str = "ensemble_mc_dropout_unet"
    ):
        super().__init__(name=name)
        self.input_size = input_size  # Store input size for building the model.
        self.n_models = n_models
        self.submodels = []

        for i in range(n_models):
            # Optionally, set a different random seed for each submodel.
            tf.keras.utils.set_random_seed(42 + i)

            # Build a submodel using MCDropoutUnetBuilder with the provided parameters.
            builder = MCDropoutUnetBuilder(
                input_size=input_size,
                num_classes=num_classes,
                num_levels=num_levels,
                num_conv_per_level=num_conv_per_level,
                num_filters=num_filters,
                regularisation=regularisation,
                use_attention=use_attention,
                activation=activation,
                return_logits=return_logits
            )
            submodel = builder.build_model()
            self.submodels.append(submodel)

    def call(self, inputs, training=None):
        # Compute predictions for each submodel, ensuring dropout is active in training mode.
        preds = [submodel(inputs, training=training) for submodel in self.submodels]
        # Stack predictions along a new axis and compute their average.
        stack_preds = tf.stack(preds, axis=0)
        avg_preds = tf.reduce_mean(stack_preds, axis=0)
        return avg_preds, preds

    def build_model(self) -> keras.Model:
        # Explicitly build the model by providing an input shape.
        self.build((None, *self.input_size))
        return self
