"""Regression tests for checkpoint resume bookkeeping in
``omnilearned.utils.restore_checkpoint`` / ``save_checkpoint``.
"""

import torch
import torch.nn as nn

from omnilearned.utils import restore_checkpoint


class _TinyModel(nn.Module):
    """Minimal stand-in for PET2: restore_checkpoint only touches
    ``.body`` / ``.classifier`` / ``.generator``."""

    def __init__(self):
        super().__init__()
        self.body = nn.Linear(2, 2)
        self.classifier = None
        self.generator = None


def _write_checkpoint(path, epoch, loss=0.5, best_loss=None, best_epoch=None):
    ckpt = {
        "body": _TinyModel().body.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    if best_loss is not None:
        ckpt["best_loss"] = best_loss
    if best_epoch is not None:
        ckpt["best_epoch"] = best_epoch
    torch.save(ckpt, path)


def test_resume_does_not_skip_an_epoch(tmp_path):
    """``save_checkpoint`` stores ``epoch + 1`` (epochs completed), so the
    resume point must be exactly that value -- the training loop is
    ``for epoch in range(epoch_init, num_epochs)``. A previous ``+ 1`` here
    silently dropped one epoch of training on every resume."""
    completed_epochs = 7
    ckpt = tmp_path / "last_model_unit.pt"
    _write_checkpoint(ckpt, epoch=completed_epochs)

    start_epoch, _best_loss, _best_epoch = restore_checkpoint(
        _TinyModel(), str(tmp_path), ckpt.name, device="cpu"
    )

    assert start_epoch == completed_epochs


def test_resume_preserves_best_loss(tmp_path):
    ckpt = tmp_path / "last_model_unit.pt"
    _write_checkpoint(ckpt, epoch=3, loss=0.9, best_loss=0.1234)

    _start_epoch, best_loss, _best_epoch = restore_checkpoint(
        _TinyModel(), str(tmp_path), ckpt.name, device="cpu"
    )

    assert best_loss == 0.1234


def test_resume_preserves_best_epoch_for_patience(tmp_path):
    """Patience is tracked as ``epoch - best_epoch``; if best_epoch is lost
    across a resume the early-stop counter silently restarts at 0."""
    ckpt = tmp_path / "last_model_unit.pt"
    _write_checkpoint(ckpt, epoch=20, best_loss=0.5, best_epoch=11)

    _start_epoch, _best_loss, best_epoch = restore_checkpoint(
        _TinyModel(), str(tmp_path), ckpt.name, device="cpu"
    )

    assert best_epoch == 11


def test_old_checkpoint_without_best_fields_still_loads(tmp_path):
    """best_model_ checkpoints written before this change carry neither
    best_loss nor best_epoch; restore must fall back, not raise."""
    ckpt = tmp_path / "best_model_unit.pt"
    _write_checkpoint(ckpt, epoch=5, loss=0.42)

    start_epoch, best_loss, best_epoch = restore_checkpoint(
        _TinyModel(), str(tmp_path), ckpt.name, device="cpu"
    )

    assert start_epoch == 5
    assert best_loss == 0.42
    assert best_epoch == 4
