import argparse
import csv
import os
from pathlib import Path

import tensorflow as tf
from keras import Model
from tensorflow.keras import optimizers, mixed_precision
from tensorflow.keras.models import load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Progbar
from tensorflow_addons.layers import GroupNormalization

from DeepLearning.config.config import load_config
from DeepLearning.dataloader.dataloader import get_dataset
from DeepLearning.losses.losses import focal_loss, ce_loss, focal_distillation_loss, kl_distillation_loss
from DeepLearning.models.custom_layers import PixelShuffle, AttentionBlock
from DeepLearning.models.unets import build_unet
from DeepLearning.processing.transforms import Transforms

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
mixed_precision.set_global_policy('mixed_float16')


def read_last_best_val_loss(log_file_path):
    """
    This method read_last_best_val_loss reads the content of a log file in CSV format and extracts the last recorded validation loss.

    Parameters:
    - log_file_path (str): The path of the log file to read.

    Returns:
    - float: The last recorded validation loss, or float('inf') if the log file is empty or doesn't exist.

    """
    try:
        with open(log_file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            last_row = None
            for row in reader:
                last_row = row  # Continuously overwrite to keep the last row
            if last_row and len(last_row) > 4:
                return float(last_row[4])  # Assuming validation loss is the fifth column
            else:
                return float('inf')
    except FileNotFoundError:
        return float('inf')


def get_last_epoch(log_file_path):
    """
    Get the number of lines in a log file.

    Args:
        log_file_path (str): Path to the log file.

    Returns:
        int: The number of lines in the log file.
        If the file is not found, returns 0.
    """
    try:
        with open(log_file_path, 'r') as file:
            return len(file.readlines())
    except FileNotFoundError:
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help=r'Config file (YAML), see example in attention_unet/config/example_config.yaml.',
        default=Path('configurations/configuration.yaml'))
    args = parser.parse_args()
    cfg = load_config(args.config)


    @tf.function(jit_compile=cfg['USE_XLA'])
    def train_step(batch, class_weights, alpha=0.01):
        def step_fn(batch, class_weights):
            x, y = batch
            with tf.GradientTape() as tape:
                # Forward pass through the teacher model
                teacher_logits = teacher_model(x, training=False)
                teacher_logits = tf.cast(teacher_logits, tf.float32)  # Old Models need this.
                # Forward pass through the student model
                y_preds, student_logits = model(x, training=True)
                if cfg['USE_FOCAL_LOSS']:
                    soft_target_loss = focal_distillation_loss(teacher_logits, student_logits)
                else:
                    soft_target_loss = kl_distillation_loss(teacher_logits, student_logits)

                # Calculate the hard label loss.
                if cfg['USE_FOCAL_LOSS']:
                    hard_label_loss = focal_loss(y_true=y, y_pred=y_preds,
                                                 alpha_weights=class_weights)
                else:
                    hard_label_loss = ce_loss(y_true=y, y_pred=y_preds,
                                              class_weight=class_weights)

                # Total loss combines soft and hard loss and is weighted by alpha
                loss = (1 - alpha) * hard_label_loss + alpha * soft_target_loss

                # Scale loss as per optimizer's requirements for mixed precision training
                scaled_loss = optimizer.get_scaled_loss(loss)

            # Calculate gradients and unscale
            scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
            gradients = optimizer.get_unscaled_gradients(scaled_gradients)

            # Apply gradients to update the model
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            return loss, (1 - alpha) * hard_label_loss, alpha * soft_target_loss

        # Distribute the step function across devices
        (
            per_replica_losses,
            per_replica_hard_losses,
            per_replica_soft_losses
        ) = strategy.run(
            step_fn,
            args=(batch, class_weights)
        )
        # Reduce the losses across replicas
        loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
        hard_label_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_hard_losses, axis=None)
        soft_target_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_soft_losses, axis=None)
        return loss, hard_label_loss, soft_target_loss


    @tf.function(jit_compile=cfg['USE_XLA'])
    def validation_step(batch=None):
        def step_fn(batch=None):
            x_val, y_val = batch
            val_probs, logits = model(x_val, training=False)
            if cfg['USE_FOCAL_LOSS']:
                val_loss = focal_loss(y_true=y_val, y_pred=val_probs, alpha_weights=class_weights)
            else:
                val_loss = ce_loss(y_true=y_val, y_pred=val_probs, class_weight=class_weights)
            return val_loss

        val_loss = strategy.run(step_fn, args=(batch,))
        per_replica_val_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, val_loss, axis=None)
        return per_replica_val_loss


    checkpoint_dir = f"{cfg['MODEL_DIR']}/{cfg['MODEL_NAME']}/"

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(checkpoint_dir + 'paras.txt', 'w') as config_log:
        print(cfg, file=config_log)

    model_save_log_path = f'{checkpoint_dir}model_save_log.txt'
    best_val_loss = read_last_best_val_loss(model_save_log_path)
    error_log_path = f'{checkpoint_dir}errors_{cfg["MODEL_NAME"][:-3]}.csv'
    start_epoch = get_last_epoch(error_log_path)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
        transforms = Transforms()
        class_weights = tf.constant(cfg['CLASS_WEIGHTS'], dtype=tf.float32)

        global_batch_size = cfg['BATCH_SIZE'] * strategy.num_replicas_in_sync
        training_ds = get_dataset(
            cfg=cfg,
            repeat=True,
            shuffle=True,
            transforms=transforms,
            filelists=cfg['TRAINING_LIST'],
            batch_size=global_batch_size
        ).with_options(options)
        distributed_training_ds = strategy.experimental_distribute_dataset(training_ds)

        global_val_batch_size = cfg['VAL_BATCH_SIZE'] * strategy.num_replicas_in_sync
        validation_ds = get_dataset(
            cfg=cfg,
            repeat=False,
            shuffle=False,
            transforms=None,
            filelists=cfg['TESTING_LIST'],
            batch_size=global_val_batch_size  # Use the larger batch size for validation
        ).with_options(options)
        distributed_validation_ds = strategy.experimental_distribute_dataset(validation_ds)

        optimizer = optimizers.Adam(learning_rate=cfg['LEARNING_RATE'])
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

        checkpoint_model = f"{checkpoint_dir}ckpt_{cfg['MODEL_NAME']}"
        best_model = f"{checkpoint_dir}best_{cfg['MODEL_NAME']}"

        # Load the teacher model
        teacher_model = load_model(
            cfg['TEACHER_MODEL'],
            custom_objects={
                'GroupNormalization': GroupNormalization(),
                'PixelShuffle': PixelShuffle,
                'AttentionBlock': AttentionBlock
            }
        )

        logits_layer = teacher_model.layers[-2].output
        teacher_model_logits = Model(inputs=teacher_model.input, outputs=logits_layer)
        teacher_model_logits.summary()

        if os.path.exists(checkpoint_model):
            tf.print(f"Restoring from model checkpoint {checkpoint_model}")
            model = load_model(
                filepath=checkpoint_model,
                compile=False,
                custom_objects={
                    'PixelShuffle': PixelShuffle,
                    'AttentionBlock': AttentionBlock
                }
            )
        else:
            tf.print('Building new model...')
            model = build_unet(
                input_size=cfg['INPUT_SIZE'],
                num_classes=cfg['OUT_CHANNELS'],
                num_levels=cfg['N_MODEL_LEVELS'],
                num_conv_per_level=cfg['N_CONV_PER_LAYER'],
                num_filters=cfg['N_FILTERS'],
                regularisation=l2(),
                use_pixel_shuffle=cfg['USE_PIXEL_SHUFFLE'],
                use_attention=cfg['USE_ATTENTION'],
                activation=tf.keras.layers.LeakyReLU(),
                return_logits=True  # Return Logits for distillation.
            )
        model.compile()
        model.summary()

    # Main training loop
    for epoch in range(start_epoch, cfg['EPOCHS']):
        ds = iter(distributed_training_ds)
        train_pb = Progbar(
            cfg['STEPS'] // global_batch_size,
            stateful_metrics=['Epoch', 'total_loss', 'focal' if cfg['USE_FOCAL_LOSS'] else 'ce_loss',
                              'distillation_loss']
        )
        avg_loss = 0.0
        avg_hard_label_loss = 0.0
        avg_soft_target_loss = 0.0
        num_batches = 0

        for _ in range(cfg['STEPS'] // global_batch_size):
            loss, hard_label_loss, soft_target_loss = train_step(
                batch=next(ds),
                class_weights=class_weights,
                alpha=cfg['DISTILLATION_ALPHA']
            )
            avg_loss += loss
            avg_hard_label_loss += hard_label_loss
            avg_soft_target_loss += soft_target_loss
            num_batches += 1
            train_pb.update(
                num_batches,
                values=[
                    ('Epoch', int(epoch) + 1),
                    ('total_loss', avg_hard_label_loss / num_batches),
                    ('focal_loss' if cfg['USE_FOCAL_LOSS'] else 'ce_loss', avg_loss / num_batches),
                    ('distillation_loss', avg_soft_target_loss / num_batches)
                ]
            )

        val_ds = iter(distributed_validation_ds)
        val_pb = Progbar(cfg['VAL_STEPS'] // global_val_batch_size, stateful_metrics=['Epoch', 'val_loss'])
        avg_val_loss = 0.0
        num_val_batches = 0

        for _ in range(cfg['VAL_STEPS'] // global_val_batch_size):
            val_loss = validation_step(next(val_ds))
            avg_val_loss += val_loss
            num_val_batches += 1
            val_pb.update(
                num_val_batches,
                values=[('Epoch', int(epoch) + 1), ('val_loss', avg_val_loss / num_val_batches)]
            )

        with open(f'{checkpoint_dir}errors_{cfg["MODEL_NAME"][:-3]}.csv', 'a') as f:
            f.write(
                f'{epoch + 1:.0},'
                f'{avg_loss.numpy() / num_batches},'
                f'{avg_hard_label_loss.numpy() / num_batches},'
                f'{avg_soft_target_loss.numpy() / num_batches},'
                f'{avg_val_loss.numpy() / num_val_batches}\n'
            )

        model.save(checkpoint_model)

        if avg_val_loss / num_val_batches < best_val_loss:
            best_val_loss = avg_val_loss / num_val_batches
            model.save(best_model)
            with open(model_save_log_path, 'a') as save_log:
                save_log.write(
                    f'Epoch {epoch + 1}: Model saved with Validation Loss: {avg_val_loss.numpy() / num_val_batches}\n')
