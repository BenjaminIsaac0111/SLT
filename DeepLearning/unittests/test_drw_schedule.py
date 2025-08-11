"""Tests for deferred re-weighting ramp schedule."""

from __future__ import annotations

import tensorflow as tf
from pytest import approx


def drw_ramp(step: int, warmup_steps: int, drw_warmup_steps: int) -> float:
    """Compute ramp value as in training schedule."""
    step_f = tf.cast(step, tf.float32)
    post = tf.nn.relu(step_f - float(warmup_steps))
    ramp = tf.minimum(1.0, post / float(drw_warmup_steps))
    return float(ramp.numpy())


def test_ramp_zero_until_warmup() -> None:
    """Ramp stays at zero during new-data warm-up."""
    assert drw_ramp(0, 5, 10) == 0.0
    assert drw_ramp(5, 5, 10) == 0.0


def test_ramp_reaches_one_after_horizon() -> None:
    """Ramp saturates at one after DRW warm-up horizon."""
    assert drw_ramp(15, 5, 10) == 1.0


def test_ramp_increases_linearly_post_warmup() -> None:
    """Ramp grows linearly after warm-up before saturation."""
    assert drw_ramp(10, 5, 10) == approx(0.5)
