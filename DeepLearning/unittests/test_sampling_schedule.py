from types import SimpleNamespace
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from DeepLearning.training.fine_tuning import Trainer


def _make_trainer(new_iter=True):
    cfg = SimpleNamespace(
        batch_size=2,
        warmup_steps=10,
        decay_schedule="half_life",
        half_life=5,
    )
    trainer = Trainer.__new__(Trainer)
    trainer.cfg = cfg
    trainer.new_iter = object() if new_iter else None
    trainer.old_size = 100
    trainer.new_size = 50
    return trainer


def test_p_new_no_new_data_returns_zero():
    trainer = _make_trainer(new_iter=False)
    assert trainer.p_new(0) == 0.0


def test_p_new_respects_warmup_and_ratio():
    trainer = _make_trainer(new_iter=True)
    ratio = trainer.cfg.batch_size * trainer.new_size / (trainer.cfg.batch_size * trainer.old_size + 1e-8)
    assert trainer.p_new(trainer.cfg.warmup_steps - 1) == 1.0
    decayed = trainer.p_new(trainer.cfg.warmup_steps + trainer.cfg.half_life * 2)
    assert ratio <= decayed <= 1.0
