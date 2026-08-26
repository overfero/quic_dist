"""Shared, transport-agnostic training utilities used across every
config-driven training module in this repo (finetune.py, rlhf.py,
distill.py, pretrain.py). NONE of this touches quic_dist's transport
(process_group.py / the Rust engine) - it's all local-to-a-rank
bookkeeping: seeding, RNG state capture, checkpoint save/resume
(trainable-params-only, so a LoRA run's checkpoint stays a few MB even
for a 27B base model - the whole point, given this project's own real
history of Kaggle disk-quota crashes from accumulating multi-GB
artifacts), and a plain JSONL experiment logger.

Deliberately excluded from this module: anything that would touch
quic_dist's send/recv/barrier path (that's real transport work, scoped
separately - communication/computation overlap needs a change in
process_group.py and likely the Rust engine's async scheduling, not a
training-loop utility).
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seeds every RNG this project's training loops actually draw
    from. Call once, early, per rank - every rank uses the SAME seed
    deliberately (reproducibility means "this exact run reproduces",
    not "each rank gets a decorrelated stream"; the dataset shuffle/
    sampling that matters for correctness is already deterministic
    per-rank via the shared, order-preserving dataset build)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state_dict() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def load_rng_state_dict(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.set_rng_state requires a CPU ByteTensor specifically - a
    # real bug found via a direct checkpoint-resume test: torch.load's
    # map_location=<cuda device> (passed by load_checkpoint() so the
    # MODEL weights land on the right GPU) relocates EVERY tensor in
    # the checkpoint dict, including this one, and set_rng_state then
    # rejects it. Force it back to CPU here regardless of what
    # map_location was used for the overall torch.load call.
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        # Same map_location relocation issue as the line above, but for
        # the per-device state LIST torch.cuda.get_rng_state_all()
        # returns - each entry needs the same .cpu() fix.
        torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda"]])


class ExperimentLogger:
    """Appends one JSON object per line - loss/perplexity/lr/timing per
    step or eval, plus a config snapshot at the start. Every rank can
    log (each record carries its own `rank`), but only the last stage
    typically has a real loss to report - middle/first stages logging
    just step/timing is fine and expected.

    Deliberately NOT a W&B/TensorBoard integration - this project has
    no reliable outbound network assumption beyond the one signaling
    URL, and JSONL is trivially `pandas.read_json(lines=True)`-able or
    greppable without any extra dependency."""

    def __init__(self, path: str | None, rank: int):
        self.path = path
        self.rank = rank
        if path:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)

    def log(self, **kwargs) -> None:
        if not self.path:
            return
        record = {"rank": self.rank, "ts": time.time(), **kwargs}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_config(self, config) -> None:
        if not self.path:
            return
        from dataclasses import asdict, is_dataclass

        cfg_dict = asdict(config) if is_dataclass(config) else dict(config)
        self.log(event="config", config=cfg_dict)


def perplexity(loss: float) -> float:
    """exp(loss) for a mean cross-entropy loss - the standard causal-LM
    perplexity definition. Guarded against overflow (an early, poorly-
    initialized or diverging run can produce a loss large enough that
    exp() overflows float64, which would otherwise crash the logging
    call instead of just reporting a very large number)."""
    import math

    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


@dataclass
class CheckpointState:
    """What a checkpoint captures, beyond the trainable weights
    themselves: enough to resume training as if it had never stopped -
    optimizer momentum/variance (AdamW needs this or the first several
    post-resume steps effectively restart warmup), RNG state (so the
    exact same data order/dropout/sampling resumes, not just "some
    order"), and dataloader position (so already-seen examples aren't
    repeated - this project's dataset builders are a single in-memory
    tensor of pre-batched steps, so "position" is just a step index)."""

    step: int
    epoch: int
    batch_index: int
    extra: dict = field(default_factory=dict)


def _trainable_state_dict(model) -> dict:
    """Only params with requires_grad=True - for every LoRA/QLoRA run
    in this repo that's the adapter weights alone (a few tens of MB at
    most, even against a 27B frozen base), not the whole model. This is
    what keeps checkpointing itself from ever being the thing that
    fills the disk - see training_utils.py's module docstring."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if _is_trainable_key(model, k)}


def _is_trainable_key(model, key: str) -> bool:
    # state_dict() keys don't carry requires_grad directly - resolve
    # against named_parameters() once per call (cheap: only ever a few
    # hundred entries for a LoRA-only trainable set).
    trainable_keys = {k for k, p in model.named_parameters() if p.requires_grad}
    return key in trainable_keys


def save_checkpoint(
    checkpoint_dir: str,
    rank: int,
    model,
    optimizer: torch.optim.Optimizer,
    state: CheckpointState,
    keep_last: int = 2,
    as_best: bool = False,
) -> str:
    """Writes `<checkpoint_dir>/rank<rank>_step<step>.pt` (the "last N"
    rotation, pruning older ones for THIS rank beyond `keep_last` - a run
    left running unattended must not be the next thing that fills the
    disk via its own checkpoints) and, when `as_best=True`, ALSO copies
    that same file to `<checkpoint_dir>/rank<rank>_best.pt` - a SEPARATE
    file outside the "last N" rotation/glob pattern, so the best-so-far
    checkpoint survives even after training has moved well past it and
    rotated the corresponding step-numbered file away. Callers decide
    "is this the best" themselves (e.g. by tracking a validation metric)
    - this function only knows how to WRITE the extra copy, not what
    "best" means. Returns the `rank<rank>_step<step>.pt` path written
    (the same as when `as_best=False`)."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt = {
        "step": state.step,
        "epoch": state.epoch,
        "batch_index": state.batch_index,
        "extra": state.extra,
        "model_state": _trainable_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "rng_state": rng_state_dict(),
    }
    path = os.path.join(checkpoint_dir, f"rank{rank}_step{state.step}.pt")
    torch.save(ckpt, path)

    if as_best:
        best_path = os.path.join(checkpoint_dir, f"rank{rank}_best.pt")
        tmp_path = best_path + ".tmp"
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, best_path)  # atomic - a reader never sees a partial "best" file

    existing = sorted(
        Path(checkpoint_dir).glob(f"rank{rank}_step*.pt"),
        key=lambda p: int(p.stem.split("step")[-1]),
    )
    for stale in existing[:-keep_last] if keep_last > 0 else []:
        stale.unlink(missing_ok=True)
    return path


def find_latest_checkpoint(checkpoint_dir: str, rank: int) -> str | None:
    if not os.path.isdir(checkpoint_dir):
        return None
    candidates = sorted(
        Path(checkpoint_dir).glob(f"rank{rank}_step*.pt"),
        key=lambda p: int(p.stem.split("step")[-1]),
    )
    return str(candidates[-1]) if candidates else None


def find_best_checkpoint(checkpoint_dir: str, rank: int) -> str | None:
    """Companion to find_latest_checkpoint() - the file save_checkpoint()
    writes when called with as_best=True. None if no checkpoint has ever
    been saved as best yet (a real, expected state early in a run, not
    an error - callers should fall back to find_latest_checkpoint() in
    that case if they need SOME checkpoint to resume from)."""
    path = Path(checkpoint_dir) / f"rank{rank}_best.pt"
    return str(path) if path.exists() else None


def load_checkpoint(
    checkpoint_dir: str,
    rank: int,
    model,
    optimizer: torch.optim.Optimizer,
    map_location=None,
) -> CheckpointState | None:
    """Returns None (a clean, expected "nothing to resume" signal, not
    an error) when no checkpoint exists for this rank - callers should
    treat that as "start fresh" rather than crashing, so the SAME
    launch command works for both a first run and a resumed one."""
    path = find_latest_checkpoint(checkpoint_dir, rank)
    if path is None:
        return None
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer_state"])
    load_rng_state_dict(ckpt["rng_state"])
    return CheckpointState(step=ckpt["step"], epoch=ckpt["epoch"], batch_index=ckpt["batch_index"], extra=ckpt.get("extra", {}))
