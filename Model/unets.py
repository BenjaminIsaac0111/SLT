from tensorflow.keras.regularizers import l2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons.layers import GroupNormalization
from .custom_layers import PixelShuffle, AttentionBlock


def build_unet(
        img_size=(1024, 512),
        num_classes=9,
        num_levels=4,
        num_conv_per_level=3,
        num_filters=64,  # Fixed number of filters
        regularisation=l2(),
        use_pixel_shuffle=True,
        use_attention=True,
        activation='tanh',
        return_logits=True
):
    skip_connections = []
    inputs = keras.Input(shape=(*img_size, 3))
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

        # Upsampling
        if use_pixel_shuffle:
            x = PixelShuffle(upscale_factor=2, name=f'decoder_pixel_shuffle_{level}')(x)
        else:
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
        logits = tf.cast(logits, dtype=tf.float32) # Cant seem to do this in the logits layer directly.
        return keras.Model(inputs=inputs, outputs=[outputs, logits])
    else:
        return keras.Model(inputs=inputs, outputs=outputs)
