from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_addons.layers import GroupNormalization
from .custom_layers import PixelShuffle, AttentionBlock


def attention_block(x, g, inter_channel, id=0):
    """
    Attention block with named layers for easier identification.
    """
    phi_g = layers.Conv2D(
        inter_channel,
        kernel_size=1,
        strides=1,
        padding='same',
        name=f'phi_g_conv_{id}'
    )(g)
    phi_g = GroupNormalization(name=f'phi_g_gn_{id}')(phi_g)

    theta_x = layers.Conv2D(
        inter_channel,
        kernel_size=1,
        strides=2,
        padding='same',
        name=f'theta_x_conv_{id}'
    )(x)
    theta_x = GroupNormalization(name=f'theta_x_gn_{id}')(theta_x)

    add_xg = layers.Add(name=f'add_xg_{id}')([theta_x, phi_g])
    add_xg = layers.LeakyReLU(name=f'add_xg_leaky_relu_{id}')(add_xg)

    psi = layers.Conv2D(
        filters=1,
        kernel_size=1,
        padding='same',
        activation='sigmoid',
        name=f'attention_psi_{id}'
    )(add_xg)

    psi = PixelShuffle(2)(psi)
    psi = layers.Multiply(name=f'attention_multiply_{id}')([psi, x])

    return psi


def res_unet(
        img_size=(1024, 512),
        num_classes=9,
        num_levels=4,
        num_conv_per_level=3,
        num_filters=64,
        regularisation=None,
        use_pixel_shuffle=True,
        use_attention=False,
        activation='tanh'
):
    """
    Implements a U-Net variant with residual connections and an optional attention mechanism.

    This model is designed for tasks like image segmentation, where it can leverage both residual connections
    for better gradient flow and attention mechanisms for focusing on relevant features.

    Args:
        img_size (tuple, optional): Size of the input image (height, width). Default is (1024, 512).
        num_classes (int, optional): Number of output classes. Default is 9.
        num_levels (int, optional): Number of levels in the U-Net. Default is 3.
        num_conv_per_level (int, optional): Number of convolutional layers per level. Default is 2.
        num_filters (int, optional): Number of filters in the convolutional layers. Default is 64.
        regularisation (regularisation layer, optional): Regularisation function applied to the convolution layers.
        Default is None.
        use_pixel_shuffle (bool, optional): Whether to use Conv2DTranspose for upsampling. Default is True.
        use_attention (bool, optional): Whether to include an attention mechanism. Default is False.
        activation (str or tensorflow activation layer, optional): Activation function for the convolution layers.
        Default is 'tanh'.

    Returns:
        keras.Model: A compiled U-Net model with the specified architecture.
    """
    skip_connections = []
    inputs = keras.Input(shape=(*img_size, 3))
    x = inputs
    x = layers.Conv2D(
        num_filters,
        kernel_size=5,
        activation=activation,
        padding='same',
        kernel_regularizer=regularisation
    )(x)
    x = GroupNormalization()(x)
    previous_block = x

    # Encoder
    for _ in range(num_levels):
        for _ in range(num_conv_per_level):
            x = layers.Conv2D(
                num_filters,
                kernel_size=5,
                activation=activation,
                padding='same',
                kernel_regularizer=regularisation
            )(x)
            x = GroupNormalization()(x)
        skip_connections.append(x)
        x = layers.AvgPool2D(2)(x)

        # Residual Connection
        res = layers.Conv2D(
            num_filters,
            kernel_size=1,
            strides=2,
            padding="same",
            kernel_regularizer=regularisation
        )(previous_block)
        x = GroupNormalization()(x)
        x = layers.Add()([res, x])
        previous_block = x

    # Middle Convolution
    x = layers.Conv2D(
        num_filters,
        kernel_size=5,
        activation=activation,
        padding='same',
        kernel_regularizer=regularisation
    )(x)
    x = GroupNormalization()(x)
    x = layers.Conv2D(
        num_filters,
        kernel_size=5,
        activation=activation,
        padding='same',
        kernel_regularizer=regularisation
    )(x)
    x = GroupNormalization()(x)

    # Decoder
    id = 0
    for _ in reversed(range(num_levels)):
        theta_x = skip_connections[-1]
        phi_g = x

        if use_pixel_shuffle:
            x = PixelShuffle(upscale_factor=2)(x)
        else:
            x = layers.UpSampling2D(2)(x)  # Not Compatible with XLA in TF==2.6

        if use_attention:
            psi = attention_block(
                x=theta_x,
                g=phi_g,
                inter_channel=num_filters // 2,
                id=id
            )
            id += 1
            x = layers.Concatenate()([psi, x])
        else:
            x = layers.Concatenate()([skip_connections[-1], x])
        skip_connections.pop()

        res = layers.Conv2D(
            num_filters,
            kernel_size=1,
            activation=activation,
            padding='same',
            kernel_regularizer=regularisation
        )(x)
        x = GroupNormalization()(x)

        for _ in range(num_conv_per_level):
            x = layers.Conv2D(
                num_filters,
                kernel_size=5,
                activation=activation,
                padding='same',
                kernel_regularizer=regularisation
            )(x)
            x = GroupNormalization()(x)
        x = layers.Add()([res, x])

    x = layers.Conv2D(
        num_classes,
        kernel_size=1,
        activation=activation,
        padding="same",
        kernel_regularizer=regularisation
    )(x)
    outputs = layers.Softmax(dtype="float32")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def res_unet_v2(
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
            x = layers.UpSampling2D(2, name=f'decoder_upsampling_{level}')(x)

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
        name=f'output_conv'
    )(x)
    outputs = layers.Softmax(dtype="float32")(logits)

    if return_logits:
        return keras.Model(inputs=inputs, outputs=logits)
    else:
        return keras.Model(inputs=inputs, outputs=outputs)
