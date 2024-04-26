import argparse
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import optimizers, mixed_precision
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Progbar
from tensorflow.keras.regularizers import l2

from Losses.losses import ce_loss
from Processing.transforms import Transforms
from Model.unets import res_unet, build_unet
from Model.custom_layers import PixelShuffle
from cfg.config import load_config
from Dataloader.dataloader import get_dataset

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
mixed_precision.set_global_policy('mixed_float16')


def read_last_best_val_loss(log_file_path):
    try:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
            if lines:
                last_line = lines[-1]
                return float(last_line.split('Validation Loss: ')[-1])
            else:
                return float('inf')
    except FileNotFoundError:
        return float('inf')


def get_last_epoch(log_file_path):
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
        help=r'Config file (YAML), see example in cfg/example_config.yaml.',
        default=Path('configurations/configuration.yaml'))
    args = parser.parse_args()
    cfg = load_config(args.config)


    @tf.function(jit_compile=cfg['USE_XLA'])
    def train_step(batch=None, class_weights=None):
        def step_fn(batch=None, class_weights=None):
            with tf.GradientTape() as tape:
                x, y = batch
                probs = model(x, training=True)
                loss = ce_loss(y_true=y, y_pred=probs, class_weight=class_weights)
                scaled_total_loss = optimizer.get_scaled_loss(loss)
            scaled_grads = tape.gradient(scaled_total_loss, model.trainable_weights)
            grads = optimizer.get_unscaled_gradients(scaled_grads)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            return loss

        pce_loss = strategy.run(step_fn, args=(batch, class_weights))
        per_replica_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, pce_loss, axis=None)
        return per_replica_loss


    @tf.function(jit_compile=cfg['USE_XLA'])
    def validation_step(batch=None):
        def step_fn(batch=None):
            x_val, y_val = batch
            val_probs = model(x_val, training=False)
            val_loss = ce_loss(y_true=y_val, y_pred=val_probs)
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
    error_log_path = f'{checkpoint_dir}errors_{cfg["MODEL_NAME"][:-3]}.txt'
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

        if os.path.exists(checkpoint_model):
            tf.print(f"Restoring from model checkpoint {checkpoint_model}")
            model = load_model(
                filepath=checkpoint_model,
                compile=False,
                custom_objects={'PixelShuffle': PixelShuffle}
            )
        else:
            tf.print('Building new model...')
            model = build_unet(
                img_size=cfg['INPUT_SIZE'],
                num_classes=cfg['OUT_CHANNELS'],
                num_levels=cfg['N_MODEL_LEVELS'],
                num_conv_per_level=cfg['N_CONV_PER_LAYER'],
                num_filters=cfg['N_FILTERS'],
                regularisation=l2(),
                use_pixel_shuffle=True,
                use_attention=cfg['USE_ATTENTION'],
                activation=tf.keras.layers.LeakyReLU(),
                return_logits=False
            )
        model.compile()
        model.summary()

    # Main training loop
    for epoch in range(start_epoch, cfg['EPOCHS']):
        ds = iter(distributed_training_ds)
        train_pb = Progbar(cfg['STEPS'] // global_batch_size, stateful_metrics=['Epoch', 'ce_loss'])
        avg_loss = 0.0
        num_batches = 0

        for _ in range(cfg['STEPS'] // global_batch_size):
            loss = train_step(next(ds), class_weights)
            avg_loss += loss
            num_batches += 1
            train_pb.update(
                num_batches,
                values=[('Epoch', int(epoch) + 1), ('ce_loss', avg_loss / num_batches)]
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

        with open(f'{checkpoint_dir}errors_{cfg["MODEL_NAME"][:-3]}.txt', 'a') as f:
            f.write(f'{avg_loss.numpy() / num_batches}\t{avg_val_loss.numpy() / num_val_batches}\n')

        model.save(checkpoint_model)

        if avg_val_loss / num_val_batches < best_val_loss:
            best_val_loss = avg_val_loss / num_val_batches
            model.save(best_model)
            with open(model_save_log_path, 'a') as save_log:
                save_log.write(
                    f'Epoch {epoch+1}: Model saved with Validation Loss: {avg_val_loss.numpy() / num_val_batches}\n')
