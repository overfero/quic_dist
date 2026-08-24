"""Correctness tests for `training_utils.py`'s checkpoint machinery
(`save_checkpoint`/`load_checkpoint`/`find_latest_checkpoint`) - pure
CPU, no signaling server/network involved (this module is explicitly
transport-agnostic, see its own module docstring), so these tests don't
use `_helpers.py`'s `SignalingServer`/`run_workers` at all, unlike every
other file in this directory."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # make `quic_dist` importable, same as _helpers.py does

import torch
import torch.nn as nn

from quic_dist.training_utils import (
    CheckpointState,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


class _TinyModel(nn.Module):
    """A frozen "base" layer plus a trainable "adapter" layer - mirrors
    the LoRA-style shape `_trainable_state_dict` is actually built for
    (most params frozen, a small trainable subset on top)."""

    def __init__(self):
        super().__init__()
        self.base = nn.Linear(4, 4)
        self.adapter = nn.Linear(4, 4)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.adapter(self.base(x))


def _build_model_and_optimizer():
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.adapter.parameters(), lr=1e-3)
    return model, optimizer


def _train_one_step(model, optimizer):
    optimizer.zero_grad()
    out = model(torch.randn(2, 4))
    out.sum().backward()
    optimizer.step()


def test_load_checkpoint_returns_none_when_nothing_to_resume(tmp_path):
    model, optimizer = _build_model_and_optimizer()
    assert load_checkpoint(str(tmp_path / "does_not_exist"), rank=0, model=model, optimizer=optimizer) is None
    # An existing-but-empty dir must behave identically - "no checkpoint
    # for this rank yet" is the same case as "the dir was never created".
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert load_checkpoint(str(empty_dir), rank=0, model=model, optimizer=optimizer) is None


def test_save_load_roundtrip_restores_trainable_params_optimizer_and_rng(tmp_path):
    set_seed(123)
    model, optimizer = _build_model_and_optimizer()
    _train_one_step(model, optimizer)  # gives the optimizer real momentum/variance state to round-trip

    saved_adapter_weight = model.adapter.weight.detach().clone()
    saved_base_weight = model.base.weight.detach().clone()
    saved_optimizer_state = {k: v for k, v in optimizer.state_dict()["state"][0].items()}

    checkpoint_dir = str(tmp_path / "ckpt")
    save_checkpoint(checkpoint_dir, rank=0, model=model, optimizer=optimizer, state=CheckpointState(step=5, epoch=1, batch_index=3))

    # A reference draw from the RNG state exactly as it was right after
    # saving - resuming must reproduce this, not just "a" random value.
    expected_next_draw = torch.rand(3)

    # Fresh model/optimizer, deliberately NOT sharing any state with the
    # ones above, and re-seeded differently so a silent no-op load
    # couldn't accidentally pass this test.
    set_seed(999)
    fresh_model, fresh_optimizer = _build_model_and_optimizer()

    resumed = load_checkpoint(checkpoint_dir, rank=0, model=fresh_model, optimizer=fresh_optimizer)

    assert resumed is not None
    assert (resumed.step, resumed.epoch, resumed.batch_index) == (5, 1, 3)

    assert torch.equal(fresh_model.adapter.weight, saved_adapter_weight), "trainable (adapter) params must be restored"

    resumed_state = optimizer_first_param_state(fresh_optimizer)
    for key, value in saved_optimizer_state.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(resumed_state[key], value), f"optimizer state {key!r} must round-trip"
        else:
            assert resumed_state[key] == value, f"optimizer state {key!r} must round-trip"

    assert torch.equal(torch.rand(3), expected_next_draw), "RNG state must be restored bit-for-bit, not just re-seeded"

    # Base layer was frozen and never in the checkpoint's model_state at
    # all - load_checkpoint's strict=False load must leave it exactly as
    # the fresh model initialized it, i.e. NOT equal to the saved run's
    # base weight (different random init, since fresh_model was built
    # under a different seed).
    assert not torch.equal(fresh_model.base.weight, saved_base_weight)


def optimizer_first_param_state(optimizer):
    return next(iter(optimizer.state_dict()["state"].values()))


def test_save_checkpoint_prunes_beyond_keep_last(tmp_path):
    model, optimizer = _build_model_and_optimizer()
    checkpoint_dir = str(tmp_path / "ckpt")

    for step in (1, 2, 3, 4):
        _train_one_step(model, optimizer)
        save_checkpoint(checkpoint_dir, rank=0, model=model, optimizer=optimizer, state=CheckpointState(step=step, epoch=0, batch_index=0), keep_last=2)

    remaining = sorted(Path(checkpoint_dir).glob("rank0_step*.pt"))
    assert len(remaining) == 2, f"expected exactly keep_last=2 checkpoints, found {remaining}"
    assert [p.name for p in remaining] == ["rank0_step3.pt", "rank0_step4.pt"], "the two most recent steps must be the ones kept"

    assert find_latest_checkpoint(checkpoint_dir, rank=0) == str(tmp_path / "ckpt" / "rank0_step4.pt")


def test_checkpoint_is_isolated_per_rank(tmp_path):
    checkpoint_dir = str(tmp_path / "ckpt")
    model0, optimizer0 = _build_model_and_optimizer()
    model1, optimizer1 = _build_model_and_optimizer()

    save_checkpoint(checkpoint_dir, rank=0, model=model0, optimizer=optimizer0, state=CheckpointState(step=1, epoch=0, batch_index=0))

    # rank 1 has never checkpointed in this dir - must still cleanly
    # report "nothing to resume", not accidentally pick up rank 0's file.
    assert find_latest_checkpoint(checkpoint_dir, rank=1) is None
    assert load_checkpoint(checkpoint_dir, rank=1, model=model1, optimizer=optimizer1) is None
