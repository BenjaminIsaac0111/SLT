import tensorflow as tf



@tf.function
def ce_loss(y_true, y_pred, class_weight=None):
    if class_weight is None:
        class_weight = tf.ones_like(y_true)
    weights = tf.reduce_sum(y_true * class_weight, axis=-1)
    ce_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred) * tf.cast(weights, tf.float32))
    return ce_loss


def focal_loss(y_true, y_pred, gamma=2.0, alpha_weights=None):
    """
    Calculate the focal loss for each pixel with class weighting and then average across all pixels.

    Args:
        y_true (tensor): True labels with shape [batch, height, width, num_classes].
        y_pred (tensor): Predictions with shape [batch, height, width, num_classes].
        gamma (float): Focusing parameter.
        alpha_weights (tensor): Class weights. It should have the same shape as the class axis of y_true/y_pred.

    Returns:
        loss (tensor): Computed focal loss value.
    """
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

    # Calculate cross entropy loss
    cross_entropy = -y_true * tf.math.log(y_pred)

    # If alpha_weights provided, use them, otherwise use the default alpha scalar.
    if alpha_weights is not None:
        alpha_factor = y_true * alpha_weights
    else:
        alpha_factor = y_true * 0.25  # default alpha scalar if no weights provided

    # Calculate focal loss components
    loss = alpha_factor * tf.math.pow(1 - y_pred, gamma) * cross_entropy

    # Sum over the class dimension and average over all other dimensions
    return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))


@tf.function
def custom_validation_iou(y_true, y_pred):
    """
    Calculate Intersection over Union for the regions specified by the one-hot encoded mask.

    Args:
        y_true: The ground truth labels, one-hot encoded.
        y_pred: The predicted logits.
    Returns:
        The average IoU score across all classes for the masked regions.
    """
    # Convert predictions from logits to one-hot format
    mask = tf.cast(y_true != 0, tf.float32)

    y_pred = tf.nn.softmax(y_pred)
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred_one_hot = tf.one_hot(y_pred, depth=tf.shape(mask)[-1], dtype=tf.float32)

    # Apply mask
    y_true_masked = y_true * mask
    y_pred_masked = y_pred_one_hot * mask

    intersection = tf.reduce_sum(y_true_masked * y_pred_masked, axis=[0, 1, 2])
    union = tf.reduce_sum(y_true_masked + y_pred_masked, axis=[0, 1, 2]) - intersection

    union = tf.where(union > 0, union, tf.ones_like(union))

    iou_scores = intersection / union
    valid_classes = tf.reduce_sum(mask, axis=[0, 1, 2]) > 0
    valid_iou_scores = tf.boolean_mask(iou_scores, valid_classes)

    average_iou = tf.reduce_mean(valid_iou_scores)
    return average_iou
