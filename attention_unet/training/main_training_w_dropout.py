import argparse
import os
import csv
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, mixed_precision
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Progbar
from tensorflow.keras.regularizers import l2

from sklearn.metrics import f1_score
from attention_unet.losses.losses import focal_loss, ce_loss
from attention_unet.processing.transforms import Transforms
from attention_unet.models.unets import build_mc_dropout_unet
from attention_unet.models.custom_layers import SpatialConcreteDropout, DropoutAttentionBlock, GroupNormalization
from attention_unet.config.config import load_config
from attention_unet.dataloader.dataloader import get_dataset

# Configure GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

mixed_precision.set_global_policy('mixed_float16')


def read_last_best_val_f1(log_file_path):
    try:
        with open(log_file_path, 'r') as file:
            lines = file.readlines()
            if lines:
                last_line = lines[-1]
                return float(last_line.split('Validation F1: ')[-1])
            else:
                return float('-inf')
    except FileNotFoundError:
        return float('-inf')


def get_last_epoch(log_file_path):
    try:
        with open(log_file_path, 'r') as file:
            return len(file.readlines())
    except FileNotFoundError:
        return 0


def gather_from_replicas(strategy, value):
    gathered_value = strategy.gather(value, axis=0)
    return gathered_value


def compute_f1_score(y_true, y_pred):
    y_true = np.squeeze(np.argmax(y_true, axis=-1))
    y_pred = np.squeeze(np.argmax(y_pred, axis=-1))
    return f1_score(y_true, y_pred, average='macro')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help=r'Config file (YAML), see example in attention_unet/config/example_config.yaml.',
        default=Path('configurations/configuration.yaml'))
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_prefix = ''


    @tf.function(jit_compile=cfg['USE_XLA'])
    def compiled_train_step(batch, class_weights):
        with tf.GradientTape() as tape:
            x, y = batch
            probs = model(x, training=True)
            loss = focal_loss(y_true=y, y_pred=probs, alpha_weights=class_weights)
            scaled_total_loss = optimizer.get_scaled_loss(loss)
        scaled_grads = tape.gradient(scaled_total_loss, model.trainable_weights)
        grads = optimizer.get_unscaled_gradients(scaled_grads)
        return loss, grads


    def train_step(batch=None, class_weights=1):
        loss, grads = strategy.run(compiled_train_step, args=(batch, class_weights))
        per_replica_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
        strategy.run(apply_gradients, args=(grads,))
        return per_replica_loss


    @tf.function
    def apply_gradients(grads):
        optimizer.apply_gradients(zip(grads, model.trainable_weights))


    @tf.function(jit_compile=cfg['USE_XLA'])
    def compiled_validation_step(batch):
        x_val, y_val = batch
        val_probs = model(x_val, training=False)
        val_loss = ce_loss(y_true=y_val, y_pred=val_probs)
        return val_loss, val_probs, y_val


    def validation_step(batch=None):
        val_loss, val_probs, y_val = strategy.run(compiled_validation_step, args=(batch,))
        per_replica_val_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, val_loss, axis=None)
        return per_replica_val_loss, val_probs, y_val


    checkpoint_dir = f"{cfg['MODEL_DIR']}/{model_prefix}_{cfg['MODEL_NAME']}/"

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    with open(checkpoint_dir + 'paras.txt', 'w') as config_log:
        print(cfg, file=config_log)
    model_save_log_path = f'{checkpoint_dir}{model_prefix}_model_save_log.txt'
    best_val_f1 = read_last_best_val_f1(model_save_log_path)
    error_log_path = f'{checkpoint_dir}{model_prefix}_errors_{cfg["MODEL_NAME"][:-3]}.txt'
    p_logit_log_path = f'{checkpoint_dir}{model_prefix}_p_logit_{cfg["MODEL_NAME"][:-3]}.csv'
    start_epoch = get_last_epoch(error_log_path)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
        transforms = Transforms()
        class_weights = tf.constant(cfg['CLASS_WEIGHTS'], dtype=tf.float32)

        global_batch_size = cfg['BATCH_SIZE'] * strategy.num_replicas_in_sync
        original_learning_rate = cfg['LEARNING_RATE']
        scaling_factor = global_batch_size / cfg['BATCH_SIZE']
        adjusted_learning_rate = original_learning_rate * scaling_factor

        global_steps = cfg['STEPS'] // cfg['BATCH_SIZE'] // strategy.num_replicas_in_sync
        training_ds = get_dataset(
            cfg=cfg,
            repeat=True,
            shuffle=True,
            transforms=transforms,
            filelists=cfg['TRAINING_LIST'],
            batch_size=global_batch_size
        ).take(global_steps).with_options(options)
        distributed_training_ds = strategy.experimental_distribute_dataset(training_ds)

        global_val_batch_size = cfg['VAL_BATCH_SIZE'] * strategy.num_replicas_in_sync
        global_val_steps = cfg['VAL_STEPS'] // cfg['BATCH_SIZE'] // strategy.num_replicas_in_sync
        validation_ds = get_dataset(
            cfg=cfg,
            repeat=False,
            shuffle=False,
            transforms=None,
            filelists=cfg['TESTING_LIST'],
            batch_size=global_val_batch_size
        ).take(global_val_steps).with_options(options)
        distributed_validation_ds = strategy.experimental_distribute_dataset(validation_ds)

        optimizer = optimizers.Adam(learning_rate=adjusted_learning_rate)
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)

        checkpoint_model = f"{checkpoint_dir}{model_prefix}_ckpt_{cfg['MODEL_NAME']}"
        best_model = f"{checkpoint_dir}{model_prefix}_best_{cfg['MODEL_NAME']}"

        if os.path.exists(checkpoint_model):
            tf.print(f"Restoring from model checkpoint {checkpoint_model}")
            model = load_model(
                filepath=checkpoint_model,
                compile=False,
                custom_objects={
                    'DropoutAttentionBlock': DropoutAttentionBlock,
                    'GroupNormalization': GroupNormalization,
                    'SpatialConcreteDropout': SpatialConcreteDropout,
                }
            )
        else:
            tf.print('Building new model...')
            model = build_mc_dropout_unet(
                input_size=cfg['INPUT_SIZE'],
                num_classes=cfg['OUT_CHANNELS'],
                num_levels=cfg['N_MODEL_LEVELS'],
                num_conv_per_level=cfg['N_CONV_PER_LAYER'],
                num_filters=cfg['N_FILTERS'],
                regularisation=l2(),
                use_attention=cfg['USE_ATTENTION'],
                activation=tf.keras.layers.LeakyReLU(),
                return_logits=False
            )
        model.compile()
        model.summary()

    if not os.path.exists(p_logit_log_path):
        with open(p_logit_log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['Epoch'] + [layer.name for layer in model.layers if isinstance(layer, SpatialConcreteDropout)]
            writer.writerow(header)

    for epoch in range(start_epoch, cfg['EPOCHS']):
        ds = iter(distributed_training_ds)
        progbar_metrics = ['Epoch', 'focal_loss' if cfg['USE_FOCAL_LOSS'] else 'ce_loss']
        train_pb = Progbar(global_steps, stateful_metrics=progbar_metrics)
        avg_loss = 0.0
        num_batches = 0
        epoch_p_logit_values = {layer.name: None for layer in model.layers if isinstance(layer, SpatialConcreteDropout)}

        for _, x, y in ds:
            loss = train_step((x, y), class_weights)
            avg_loss += loss
            num_batches += 1

            for layer in model.layers:
                if isinstance(layer, SpatialConcreteDropout):
                    epoch_p_logit_values[layer.name] = layer.p_logit.numpy().mean()

            progbar_values = [
                ('Epoch', int(epoch) + 1),
                ('focal_loss' if cfg['USE_FOCAL_LOSS'] else 'ce_loss', avg_loss / num_batches),
            ]

            train_pb.update(num_batches, values=progbar_values)

        val_ds = iter(distributed_validation_ds)
        val_pb = Progbar(global_val_steps, stateful_metrics=progbar_metrics)
        avg_val_loss = 0.0
        num_val_batches = 0
        val_f1_scores = []
        y_true_gathered = []
        y_probs_gathered = []

        for _, yx, yy in val_ds:
            val_loss, val_probs, y_true = validation_step((yx, yy))
            center_x, center_y = cfg['INPUT_SIZE'][0] // 2, cfg['INPUT_SIZE'][1] // 2
            avg_val_loss += val_loss
            num_val_batches += 1
            y_true_gathered.append(gather_from_replicas(strategy, y_true)[:, center_x, center_y, :])
            y_probs_gathered.append(gather_from_replicas(strategy, val_probs)[:, center_x, center_y, :])
            val_pb.update(
                num_val_batches,
                values=[
                    ('Epoch', int(epoch) + 1),
                    ('val_loss', avg_val_loss / num_val_batches),
                ]
            )

        # Save p_logit values to CSV
        with open(p_logit_log_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = [epoch + 1] + [epoch_p_logit_values[layer.name] for layer in model.layers if
                                 isinstance(layer, SpatialConcreteDropout)]
            writer.writerow(row)

        final_f1_score = compute_f1_score(np.array([y[0, :] for y in y_true_gathered]),
                                          np.array([y[0, :] for y in y_probs_gathered]))
        with open(f'{checkpoint_dir}{model_prefix}_errors_{cfg["MODEL_NAME"][:-3]}.txt', 'a') as f:
            f.write(
                f'{avg_loss.numpy() / num_batches}\t'
                f'{avg_val_loss.numpy() / num_val_batches}\t'
                f'{final_f1_score}\n'
            )

        model.save(checkpoint_model)

        if final_f1_score > best_val_f1:
            best_val_f1 = final_f1_score
            model.save(best_model)
            with open(model_save_log_path, 'a') as save_log:
                save_log.write(
                    f'Epoch {epoch + 1}: Model saved with Validation F1: {best_val_f1}\n')
