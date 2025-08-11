import tensorflow as tf


def ce_loss(y_true, y_pred, class_weight=None):
    """

    This method computes the cross-entropy loss between the true labels `y_true` and the predicted labels `y_pred`.
    It supports the option to apply class weights to the loss calculation.

    Parameters: - y_true: A tensor containing the true labels. Shape: [batch_size, num_classes] - y_pred: A tensor
    containing the predicted labels. Shape: [batch_size, num_classes] - class_weight (optional): A tensor specifying
    the weight for each class. If not provided, all classes will have equal weight. Shape: [batch_size, num_classes]

    Returns:
        - ce_loss: A scalar tensor representing the cross-entropy loss.

    Example usage:
        y_true = [[0, 1, 0], [1, 0, 0]]
        y_pred = [[0.3, 0.6, 0.1], [0.7, 0.2, 0.1]]
        loss = ce_loss(y_true, y_pred)
    """
    if class_weight is None:
        class_weight = tf.ones_like(y_true)
    weights = tf.reduce_sum(y_true * class_weight, axis=-1)
    ce_loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true, y_pred) * tf.cast(weights, tf.float32))
    return ce_loss


def focal_loss(y_true, y_pred, gamma: float = 2.0, alpha_weights=None):
    """Compute focal loss over *labeled* pixels only.

    The loss is masked to ignore unlabeled pixels (where the one-hot target sums
    to zero) and normalized by the count of labeled pixels in each patch.
    """
    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    cross_entropy = -y_true * tf.math.log(y_pred)

    if alpha_weights is not None:
        alpha_factor = y_true * alpha_weights
    else:
        alpha_factor = 1.0

    loss = alpha_factor * tf.math.pow(1.0 - y_pred, gamma) * cross_entropy
    loss = tf.reduce_sum(loss, axis=-1)  # [B,H,W]

    mask = tf.reduce_sum(y_true, axis=-1)  # [B,H,W]; 1 for labeled pixels
    loss = loss * mask

    per_example = tf.reduce_sum(loss, axis=[1, 2])
    counts = tf.reduce_sum(mask, axis=[1, 2])
    loss = tf.math.divide_no_nan(per_example, counts)
    return tf.reduce_mean(loss)


def kl_distillation_loss(teacher_logits, student_logits, temperature=2.0):
    """
    Calculates the Kullback-Leibler divergence loss between teacher and student logits for knowledge distillation.

    Parameters:
    - teacher_logits: tf.Tensor, teacher model logits.
    - student_logits: tf.Tensor, student model logits.
    - temperature: float, temperature factor for adjusting the soft loss (default=2.0).

    Returns:
    - soft_target_loss: tf.Tensor, the calculated soft target loss.

    """
    # Calculate the soft loss with the temperature factor adjusted
    teacher_soft_targets = tf.nn.softmax(teacher_logits / temperature)
    student_soft_predictions = tf.nn.softmax(student_logits / temperature)
    soft_target_loss = tf.keras.losses.kullback_leibler_divergence(
        y_true=teacher_soft_targets,
        y_pred=student_soft_predictions
    ) * (temperature ** 2)  # Note the squaring of the temperature
    return soft_target_loss


def focal_distillation_loss(teacher_logits, student_logits, gamma=2.0, alpha_weights=None, temperature=2.0):
    """

    Calculate the focal distillation loss between the teacher and student logits.

    Parameters:
    teacher_logits (tf.Tensor): The logits from the teacher model.
    student_logits (tf.Tensor): The logits from the student model.
    gamma (float, optional): The focusing parameter. Defaults to 2.0.
    alpha_weights (tf.Tensor, optional): The weights for each class. Defaults to None.
    temperature (float, optional): The temperature for the softmax operation. Defaults to 2.0.

    Returns:
    tf.Tensor: The focal distillation loss.

    """

    teacher_soft_labels = tf.nn.softmax(teacher_logits / temperature)
    student_soft_labels = tf.nn.softmax(student_logits / temperature)
    # Calculate cross entropy loss
    kl_divergence = student_soft_labels * tf.math.log(student_soft_labels / teacher_soft_labels)

    # Calculate focal loss components
    modulating_factor = tf.math.pow(1 - tf.abs(student_soft_labels - teacher_soft_labels), gamma) * kl_divergence
    # If alpha_weights provided, multiply by class weights
    if alpha_weights is not None:
        modulating_factor *= alpha_weights
    # Sum over the class dimension and average over all other dimensions
    return tf.reduce_mean(tf.reduce_sum(modulating_factor, axis=-1))


@tf.function
def custom_validation_iou(y_true, y_pred):
    """
    Calculates the Average Intersection over Union (IoU) score for semantic segmentation models.
    Nullifies contributions from location without ground truth in order to calculate IOU on areas with ground truth.
    Args:
        y_true: The true labels of the segmentation masks. Expected shape is (batch_size, height, width, num_classes).
        y_pred: The predicted logits of the segmentation masks. Expected shape is (batch_size, height, width, num_classes).

    Returns:
        The average IoU score for the valid classes.

    Raises:
        None

    Example:
        y_true = tf.constant([[[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                               [[1, 1, 1], [0, 0, 0], [1, 1, 1]],
                               [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]])
        y_pred = tf.constant([[[[0.9, 0.1, 0], [0.1, 0.8, 0.1], [0, 0, 0.9]],
                               [[0.6, 0.3, 0.1], [0.1, 0.2, 0.7], [0.7, 0.2, 0.1]],
                               [[0, 0, 0], [0, 0.9, 0], [0, 0, 0]]]])

        result = custom_validation_iou(y_true, y_pred)
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
