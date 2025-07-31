import pytest

# Skip tests if TensorFlow is not installed
tf = pytest.importorskip('tensorflow')


def test_checkpoint_restores_training_state(tmp_path):
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    optimizer = tf.keras.optimizers.SGD()
    step = tf.Variable(0, dtype=tf.int64)
    epoch = tf.Variable(0, dtype=tf.int64)
    best = tf.Variable(float("-inf"), dtype=tf.float32)
    baseline = tf.Variable(float("-inf"), dtype=tf.float32)
    ckpt = tf.train.Checkpoint(
        step=step,
        epoch=epoch,
        optimizer=optimizer,
        model=model,
        best_val_f1=best,
        baseline_val_f1=baseline,
    )
    manager = tf.train.CheckpointManager(ckpt, tmp_path, max_to_keep=1)

    best.assign(0.5)
    baseline.assign(0.4)
    step.assign(10)
    epoch.assign(2)
    manager.save()

    new_model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    new_optimizer = tf.keras.optimizers.SGD()
    new_step = tf.Variable(0, dtype=tf.int64)
    new_epoch = tf.Variable(0, dtype=tf.int64)
    new_best = tf.Variable(float("-inf"), dtype=tf.float32)
    new_baseline = tf.Variable(float("-inf"), dtype=tf.float32)
    new_ckpt = tf.train.Checkpoint(
        step=new_step,
        epoch=new_epoch,
        optimizer=new_optimizer,
        model=new_model,
        best_val_f1=new_best,
        baseline_val_f1=new_baseline,
    )
    new_ckpt.restore(manager.latest_checkpoint).expect_partial()

    assert new_step.numpy() == 10
    assert new_epoch.numpy() == 2
    assert new_best.numpy() == pytest.approx(0.5)
    assert new_baseline.numpy() == pytest.approx(0.4)
