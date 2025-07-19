import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.layers.experimental import preprocessing


class Transforms(tf.keras.layers.Layer):
    def __init__(self, seed=None):
        """
        Initializes an instance of the Transforms class.

        Parameters: seed (int, optional): Integer seed value for random number generation. If provided, it ensures
        reproducibility of the transformations applied. If not specified, the default seed behavior of TensorFlow is
        used.
        """
        super().__init__()

        # Geometric transforms. Make sure these are applied to ground truth masks.
        self.transforms = tf.keras.Sequential(
            [
                preprocessing.RandomFlip(seed=seed),
                preprocessing.RandomRotation(factor=10, seed=seed),
                preprocessing.RandomTranslation(height_factor=(-1, 1), width_factor=(-1, 1))
            ]
        )
        # Transforms that should not affect the ground truth mask.
        self.destructive_transforms = tf.keras.Sequential(
            [
                DestructiveAugmentor(seed=seed)
            ]
        )

    def call(self, x, y, **kwargs):
        """
        Applies the defined transforms to the input data.

        Parameters:
            x (tf.Tensor): Input image tensor.
            y (tf.Tensor): Ground truth mask tensor.
            **kwargs: Additional keyword arguments (not used in this method).

        Returns:
            tf.Tensor: Transformed image tensor (x) after applying destructive transforms.
            tf.Tensor: Transformed mask tensor (y) after applying geometric transforms.
        """
        # Concatenate image and ground truth mask concat in order to use apply the same transforms to both mask and
        # image.
        if mixed_precision.global_policy().name == 'mixed_float16':
            x = tf.cast(x, tf.float32)
        aug = self.transforms(tf.concat([x, y], axis=3))
        x, y = tf.split(aug, [x.shape[-1], y.shape[-1]], axis=3)

        # Transforms that only change the image, not the mask else we destroy the labelling.
        x = self.destructive_transforms(x)
        return x, tf.cast(y, tf.float32)


class Augmentation(tf.keras.layers.Layer):
    """
    Base augmentation class.

    Base augmentation class. Contains the random_execute method.

    Methods:
        random_execute: method that returns true or false based on a probability. Used to determine whether an
    augmentation will be run.
    """

    def __init__(self, seed=None):
        super().__init__()
        self.seed = seed

    def random_execute(self, prob: float) -> bool:
        """
        random_execute function.

        Arguments:
            prob: a float value from 0-1 that determines the probability.

        Returns:
            returns true or false based on the probability.
        """

        return tf.random.uniform([], minval=0, maxval=1, seed=self.seed) < prob


class DestructiveAugmentor(tf.keras.Model):
    """
    RandomAugmentor layer.

    RandomAugmentor class. Chains all the augmentations into
    one pipeline.

    Attributes:
        random_color_jitter: Instance variable representing the RandomColorJitter layer.
        random_blur: Instance variable representing the RandomBlur layer
        random_solarize: Instance variable representing the RandomSolarize layer

    Methods:
        call: chains layers in pipeline together
    """

    def __init__(self, seed: int):
        """
        Initializes an instance of the DestructiveAugmentor class.

        Parameters:
            seed (int): Integer seed value for random number generation. It ensures
            reproducibility of the augmentation operations.
        """
        super(DestructiveAugmentor, self).__init__()
        self.seed = seed
        self.random_color_jitter = RandomColorJitter(seed=self.seed)
        self.random_greyscale = RandomGreyScale(seed=self.seed)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Applies the defined augmentation operations to the input tensor.

        Parameters:
            inputs (tf.Tensor): Input image tensor.

        Returns:
            tf.Tensor: Transformed image tensor after applying the augmentation operations.
        """
        inputs = self.random_color_jitter(inputs)
        inputs = self.random_greyscale(inputs)
        inputs = tf.clip_by_value(inputs, 0, 1)
        return inputs


class RandomColorJitter(Augmentation):
    """
    RandomColorJitter class.

    RandomColorJitter class. Randomly adds color jitter to an image.
    Color jitter means to add random brightness, contrast,
    saturation, and hue to an image. There is a 80% chance that an
    image will be randomly color-jittered.

    Methods:
        call: method that color-jitters an image 70% of
          the time.
    """

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        call function.

        Adds color jitter to image, including:
        Brightness change by a max-delta of 0.8
        Contrast change by a max-delta of 0.8
        Saturation change by a max-delta of 0.8
        Hue change by a max-delta of 0.2

        Arguments:
            x: a tf.Tensor representing the image.

        Returns:
            returns a color-jittered version of the image 80% of the time and the original image 20% of the time.
        """

        if self.random_execute(0.8):
            # Random brightness adjustment
            x = tf.image.random_brightness(x, max_delta=0.2)  # +/- 20% brightness

            # Random contrast adjustment
            x = tf.image.random_contrast(x, lower=0.85, upper=1.15)  # +/- 15% contrast

            # Random saturation adjustment
            x = tf.image.random_saturation(x, lower=0.75, upper=1.25)  # +/- 25% saturation

            # Random hue adjustment
            x = tf.image.random_hue(x, max_delta=0.05)
        return x


class RandomGreyScale(Augmentation):
    """RandomGreyScale class.

    RandomGreyScale class. Randomly greyscale an image.

    Methods:
        call: method that does random Grey Scaling 20% of the time.
    """

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """call function.

        Randomly greyscale the image.

        Arguments:
            x: a tf.Tensor representing the image.

        Returns:
            returns a greyscale version of the image 30% of the time
              and the original image 70% of the time.
        """

        if self.random_execute(0.3):
            x = tf.image.rgb_to_grayscale(x)
            x = tf.image.grayscale_to_rgb(x)
            return x
        return x
