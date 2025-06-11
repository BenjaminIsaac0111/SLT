import tensorflow as tf
from keras import Model, Input
from keras.layers import Softmax, Conv2D, Add, Concatenate, UpSampling2D, AvgPool2D
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

from .custom_layers import AttentionBlock, SpatialConcreteDropout, DropoutAttentionBlock, GroupNormalization


# ---
# NOTE: This is legacy code for the U-Net models and will be removed eventually.
# New versions are found in Model/models.py
# ---
def build_unet(
        input_size=(1024, 512, 3),
        num_classes=9,
        num_levels=4,
        num_conv_per_level=3,
        num_filters=64,  # Fixed number of filters
        regularisation=l2(),
        use_attention=True,
        activation='tanh',
        return_logits=True
):
    skip_connections = []
    inputs = keras.Input(shape=input_size)
    x = inputs

    # Initial Convolution
    x = layers.Conv2D(
        num_filters,
        kernel_size=5,
        activation=activation,
        padding='same',
        kernel_regularizer=regularisation,
        name='initial_conv'
    )(x)
    x = GroupNormalization(name='initial_gn')(x)
    previous_block_output = x

    # Encoder
    for level in range(num_levels):
        for conv in range(num_conv_per_level):
            x = layers.Conv2D(
                num_filters,
                kernel_size=5,
                activation=activation,
                padding='same',
                kernel_regularizer=regularisation,
                name=f'encoder_level_{level}_conv_{conv}'
            )(x)
            x = GroupNormalization(name=f'encoder_level_{level}_gn_{conv}')(x)
        skip_connections.append(x)
        x = layers.AvgPool2D(2, name=f'encoder_level_{level}_pooling')(x)

        # Residual connection
        residual_connection = layers.Conv2D(
            num_filters,
            kernel_size=1,
            strides=2,
            padding='same',
            kernel_regularizer=regularisation,
            name=f'encoder_residual_connection_{level}'
        )(previous_block_output)
        x = GroupNormalization()(x)
        x = layers.Add(name=f'residual_add_{level}')([x, residual_connection])
        previous_block_output = x

    # Middle
    x = layers.Conv2D(
        num_filters,
        kernel_size=5,
        activation=activation,
        padding='same',
        kernel_regularizer=regularisation,
        name='middle_conv_0'
    )(x)
    x = GroupNormalization(name='middle_gn_0')(x)
    x = layers.Conv2D(
        num_filters,
        kernel_size=5,
        activation=activation,
        padding='same',
        kernel_regularizer=regularisation,
        name='middle_conv_1'
    )(x)
    x = GroupNormalization(name='middle_gn_1')(x)

    # Decoder
    for level in reversed(range(num_levels)):
        theta_x = skip_connections[-1]
        phi_g = x

        x = layers.UpSampling2D(2, name=f'decoder_upsampling_{level}', interpolation='bilinear')(x)

        # Attention mechanism
        if use_attention:
            psi = AttentionBlock(
                inter_channel=num_filters // 2,
                name=f'attention_block_{level}'
            )([theta_x, phi_g])

            # Concatenation with skip connection
            x = layers.Concatenate(name=f'decoder_concat_{level}')([x, psi])
        else:
            x = layers.Concatenate()([skip_connections[-1], x])
        skip_connections.pop()

        residual_connection = layers.Conv2D(
            num_filters,
            kernel_size=1,
            activation=activation,
            padding='same',
            kernel_regularizer=regularisation,
            name=f'decoder_residual_connection_{level}'
        )(x)
        x = GroupNormalization()(x)

        # Convolutional layers after concatenation
        for conv in range(num_conv_per_level):
            x = layers.Conv2D(
                num_filters,
                kernel_size=5,
                activation=activation,
                padding='same',
                kernel_regularizer=regularisation,
                name=f'decoder_level_{level}_conv_{conv}'
            )(x)
            x = GroupNormalization(name=f'decoder_level_{level}_gn_{conv}')(x)
        x = layers.Add(name=f'decoder_residual_add_{level}')([x, residual_connection])

    # Output Convolution
    logits = layers.Conv2D(
        num_classes,
        kernel_size=1,
        activation=activation,
        padding="same",
        kernel_regularizer=regularisation,
        name='output_conv'
    )(x)
    outputs = layers.Softmax(dtype="float32")(logits)

    if return_logits:
        logits = tf.cast(logits, dtype=tf.float32)  # Cant seem to do this in the logits layer directly.
        return keras.Model(inputs=inputs, outputs=[outputs, logits])
    else:
        return keras.Model(inputs=inputs, outputs=outputs)


def build_mc_dropout_unet(
        input_size=(1024, 512, 3),
        num_classes=9,
        num_levels=4,
        num_conv_per_level=3,
        num_filters=64,  # Fixed number of filters
        regularisation=None,
        use_attention=True,
        activation=tf.keras.layers.LeakyReLU(),
        return_logits=False,
):
    skip_connections = []
    inputs = Input(shape=input_size, name='input_layer')
    x = inputs

    # Initial Convolution
    x = Conv2D(
        num_filters,
        kernel_size=5,
        activation=activation,
        padding='same',
        kernel_regularizer=regularisation,
        name='initial_conv'
    )(x)
    x = GroupNormalization(name='initial_gn')(x)
    previous_block_output = x

    # Encoder
    for level in range(num_levels):
        for conv in range(num_conv_per_level):
            x = SpatialConcreteDropout(name=f'encoder_level_{level}_spatial_dropout_{conv}', layer=Conv2D(
                num_filters,
                kernel_size=5,
                activation=activation,
                padding='same',
                kernel_regularizer=regularisation,
                name=f'encoder_level_{level}_conv_{conv}'
            ))(x)
            x = GroupNormalization(name=f'encoder_level_{level}_gn_{conv}')(x)
        skip_connections.append(x)
        x = AvgPool2D(2, name=f'encoder_level_{level}_pooling')(x)

        # Residual connection
        residual_connection = SpatialConcreteDropout(name=f'encoder_residual_level_{level}_spatial_dropout_{conv}',
                                                     layer=Conv2D(
                                                         num_filters,
                                                         kernel_size=1,
                                                         strides=2,
                                                         padding='same',
                                                         kernel_regularizer=regularisation,
                                                         name=f'encoder_residual_connection_{level}'
                                                     ))(previous_block_output)
        x = GroupNormalization(name=f'encoder_residual_gn_{level}')(x)
        x = Add(name=f'residual_add_{level}')([x, residual_connection])
        previous_block_output = x

    # Middle
    x = SpatialConcreteDropout(name='middle_spatial_dropout_0', layer=Conv2D(
        num_filters,
        kernel_size=5,
        activation=activation,
        padding='same',
        kernel_regularizer=regularisation,
        name='middle_conv_0'
    ))(x)
    x = GroupNormalization(name='middle_gn_0')(x)
    x = SpatialConcreteDropout(name='middle_spatial_dropout_1', layer=Conv2D(
        num_filters,
        kernel_size=5,
        activation=activation,
        padding='same',
        kernel_regularizer=regularisation,
        name='middle_conv_1'
    ))(x)
    x = GroupNormalization(name='middle_gn_1')(x)

    # Decoder
    for level in reversed(range(num_levels)):
        theta_x = skip_connections[-1]
        phi_g = x

        x = UpSampling2D(2, name=f'decoder_upsampling_{level}', interpolation='bilinear')(x)

        # Attention mechanism
        if use_attention:
            psi = DropoutAttentionBlock(
                inter_channel=num_filters // 2,
                name=f'attention_block_{level}'
            )([theta_x, phi_g])

            # Concatenation with skip connection
            x = Concatenate(name=f'decoder_concat_{level}')([x, psi])
        else:
            x = Concatenate(name=f'decoder_concat_{level}')([skip_connections[-1], x])
        skip_connections.pop()

        residual_connection = SpatialConcreteDropout(name=f'decoder_residual_level_{level}_spatial_dropout_{conv}',
                                                     layer=Conv2D(
                                                         num_filters,
                                                         kernel_size=1,
                                                         activation=activation,
                                                         padding='same',
                                                         kernel_regularizer=regularisation,
                                                         name=f'decoder_residual_connection_{level}'
                                                     ))(x)
        x = GroupNormalization(name=f'decoder_residual_gn_{level}')(x)

        # Convolutional layers after concatenation
        for conv in range(num_conv_per_level):
            x = SpatialConcreteDropout(name=f'decoder_level_{level}_spatial_dropout_{conv}', layer=Conv2D(
                num_filters,
                kernel_size=5,
                activation=activation,
                padding='same',
                kernel_regularizer=regularisation,
                name=f'decoder_level_{level}_conv_{conv}'
            ))(x)
            x = GroupNormalization(name=f'decoder_level_{level}_gn_{conv}')(x)
        x = Add(name=f'decoder_residual_add_{level}')([x, residual_connection])

    # Output Convolution
    logits = Conv2D(
        num_classes,
        kernel_size=1,
        activation=activation,
        padding="same",
        kernel_regularizer=regularisation,
        name='output_conv'
    )(x)
    outputs = Softmax(dtype="float32", name='softmax_output')(logits)

    if return_logits:
        logits = tf.cast(logits, dtype=tf.float32, name='mc_unet')
        return Model(inputs=inputs, outputs=[logits])
    else:
        return Model(inputs=inputs, outputs=outputs, name='mc_unet')
