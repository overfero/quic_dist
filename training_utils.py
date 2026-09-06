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


def _tpu_chip_count() -> int:
    """Total local TPU chips on this host, from TPU_CHIPS_PER_HOST_BOUNDS
    (e.g. "2,4,1" on a v5e-8 -> 8). Falls back to 8 (this project's only
    validated TPU shape so far) if the env var isn't set - see
    resolve_device()'s docstring for where that var comes from."""
    bounds = os.environ.get("TPU_CHIPS_PER_HOST_BOUNDS")
    if not bounds:
        return 8
    dims = [int(x) for x in bounds.split(",")]
    n = 1
    for d in dims:
        n *= d
    return n


def _factor_chip_bounds(n: int) -> str:
    """A valid `TPU_CHIPS_PER_PROCESS_BOUNDS` string for an n-chip
    AXIS-ALIGNED RECTANGLE of this host's real physical topology
    (TPU_CHIPS_PER_HOST_BOUNDS, e.g. "2,4,1" on a v5e-8 - a 2x4 grid).
    Real bug this fixes: the original implementation always used the
    flat `f"{n},1,1"` shape, which is only a valid rectangle when n <=
    the physical x-dimension (2 here) - for any larger n (e.g. 6, or 8
    for the whole host) that shape doesn't decompose the real topology
    at all and crashes identically to the whole-host "8,1,1" bug this
    module's own resolve_device docstring already documents. Confirmed
    directly: n=6 as "2,3,1" (2 in x, 3 in y) lets a torch_xla process
    and a concurrent JAX process each successfully claim their own
    real, disjoint chip rectangle (e.g. chips [2,8) as "2,3,1" here,
    chips [0,2) as "2,1,1" in a sibling process) - literal "N,1,1" for
    N=6 does not.

    Only 1/2/3/4/6/8 are real rectangle sizes on a 2x4 grid (7 is prime
    and has no factor pair (a<=2, b<=4) - a real hardware constraint,
    not a limitation of this function - see this module's own
    docstring for the concrete reasoning)."""
    bounds = os.environ.get("TPU_CHIPS_PER_HOST_BOUNDS")
    if bounds:
        px, py = (int(x) for x in bounds.split(",")[:2])
    else:
        px, py = 2, 4  # this project's only validated TPU shape so far (v5e-8)
    for a in range(min(px, n), 0, -1):
        if n % a == 0 and (n // a) <= py:
            return f"{a},{n // a},1"
    return f"{n},1,1"  # no valid rectangle exists for this n - falls back to the old
                        # (known-unsafe for n>px) shape rather than raising, so a genuinely
                        # invalid n (e.g. 7) still gets a clear runtime error from libtpu
                        # itself rather than an opaque one from this function


def resolve_device(rank: int, local_index: int | None = None, tensor_parallel_size: int = 1, chip_offset: int = 0) -> torch.device:
    """Picks this rank's real accelerator device: TPU > CUDA > CPU,
    mirroring the old hardcoded `torch.device(f"cuda:{rank %
    torch.cuda.device_count()}")` pattern every *_pipeline_rank.py
    caller used, but for whichever backend is actually present.

    TPU (PJRT_DEVICE=TPU in the environment - the standard Kaggle/GCE
    TPU VM setup): each quic_dist rank is a genuinely separate OS
    process (unlike torch_xla's own xmp.spawn(), which coordinates
    sibling processes through one parent). Confirmed via a direct
    two-process test that the PJRT TPU runtime is exclusive per host by
    default - a second process's own device init hits a real `Device or
    resource busy` on /dev/vfio and aborts. TPU_VISIBLE_CHIPS (which
    physical chip(s) this process may see) plus TPU_CHIPS_PER_PROCESS_BOUNDS/
    TPU_PROCESS_BOUNDS (this process owns exactly its own chips, no
    others) fixes that - confirmed via the same test running both
    processes concurrently without conflict once these were set. Set via
    `os.environ.setdefault` (not unconditional) so a caller that already
    exports these itself - e.g. a real multi-host TPU pod launcher -
    isn't overridden. Must happen before `torch_xla` first touches the
    runtime, which is why this function imports it lazily rather than at
    module scope.

    tensor_parallel_size > 1: this rank now needs a CONTIGUOUS BLOCK of
    that many chips (not just one) - real intra-stage tensor parallelism
    via torch_xla SPMD (see tensor_parallel.py) shards weights across
    every chip a single quic_dist rank owns, so those chips must all be
    visible to (and only to) this one process, exactly like the
    single-chip case above just with a wider block. Enables SPMD mode
    (`xr.use_spmd()`) globally for this process - do not mix a
    tensor_parallel_size>1 rank with plain (non-SPMD) tensor ops in the
    same process afterward. rank=0's block is chips [0, tp_size); rank=1's
    is [tp_size, 2*tp_size); etc. - matches PipelineConfig.world_size *
    tensor_parallel_size <= total host chips (checked here, not silently
    truncated - an oversubscribed block is a real config error, not
    something to paper over by wrapping/reusing chips another rank
    already owns). The whole-host case (world_size==1,
    tensor_parallel_size==total chip count - this project's only
    LIVE-VALIDATED TP shape so far) skips TPU_VISIBLE_CHIPS/*_BOUNDS
    entirely rather than setting an arbitrary "{tp_size},1,1" - see the
    real crash this works around in the code below. A STRICT-SUBSET
    block now uses `_factor_chip_bounds()` (a real axis-aligned-rectangle
    factorization of the host's physical topology, not the old flat
    "{tp_size},1,1" shape - see that function's own docstring for the
    real crash class it fixes) - confirmed directly: a torch_xla process
    claiming a 6-chip block (chips [2,8), bounds "2,3,1") and a
    concurrent JAX process claiming a disjoint 2-chip block (chips [0,2),
    bounds "2,1,1") both succeed at the same time.

    `chip_offset`: shifts the whole block by this many chips - e.g. a
    caller that's dedicating chips [0, k) of the host to a DIFFERENT,
    concurrently-running process (a same-host inference engine, say) and
    wants quic_dist's own rank(s) confined to the REMAINING [k, total)
    chips passes `chip_offset=k`. 0 (default) = unchanged behavior,
    blocks start at chip 0 as before.

    CUDA: unchanged behavior - `cuda:{local_index}`.

    CPU: last resort, e.g. local dev/test without an accelerator."""
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        if tensor_parallel_size > 1:
            chip_count = _tpu_chip_count()
            start = chip_offset + (local_index if local_index is not None else rank) * tensor_parallel_size
            if start + tensor_parallel_size > chip_count:
                raise ValueError(
                    f"resolve_device: rank {rank} needs chips [{start}, {start + tensor_parallel_size}) "
                    f"but only {chip_count} are available on this host - chip_offset + world_size * "
                    f"tensor_parallel_size must fit within the host's total TPU chip count."
                )
            # Only restrict TPU_VISIBLE_CHIPS/*_BOUNDS when this rank owns a
            # STRICT SUBSET of the host's chips (world_size>1, combined
            # PP+TP). A real crash found running this for real: when this
            # rank claims the WHOLE host (the common tensor_parallel_size ==
            # chip_count, world_size==1 case), an arbitrary bounds
            # factorization like "8,1,1" does NOT validly decompose the
            # REAL physical topology (TPU_CHIPS_PER_HOST_BOUNDS, e.g.
            # "2,4,1" on a v5e-8 - dimension 0 there physically holds only 2
            # chips, not 8) - libtpu fatal-exits (silent `exit(1)`, no
            # Python traceback - not even a catchable exception) the moment
            # anything queries the real device topology (confirmed via a
            # direct isolated repro: xm.xla_device() itself succeeds
            # lazily, but xr.addressable_runtime_device_count() - which
            # forces real topology negotiation - is what actually crashes).
            # Skipping the restriction entirely for the whole-host case
            # sidesteps this completely: the default (no TPU_VISIBLE_CHIPS)
            # behavior already gives full, correctly-bounded visibility.
            if start != 0 or tensor_parallel_size != chip_count:
                chips = ",".join(str(c) for c in range(start, start + tensor_parallel_size))
                os.environ.setdefault("TPU_VISIBLE_CHIPS", chips)
                os.environ.setdefault("TPU_CHIPS_PER_PROCESS_BOUNDS", _factor_chip_bounds(tensor_parallel_size))
                os.environ.setdefault("TPU_PROCESS_BOUNDS", "1,1,1")
            import torch_xla.runtime as xr

            xr.use_spmd()
            import torch_xla.core.xla_model as xm

            return xm.xla_device()
        if local_index is None:
            local_index = rank % _tpu_chip_count()
        os.environ.setdefault("TPU_VISIBLE_CHIPS", str(local_index))
        os.environ.setdefault("TPU_CHIPS_PER_PROCESS_BOUNDS", "1,1,1")
        os.environ.setdefault("TPU_PROCESS_BOUNDS", "1,1,1")
        import torch_xla.core.xla_model as xm

        return xm.xla_device()
    if torch.cuda.is_available():
        if local_index is None:
            local_index = rank % torch.cuda.device_count()
        return torch.device(f"cuda:{local_index}")
    return torch.device("cpu")


def mark_step(device: torch.device) -> None:
    """No-op on CUDA/CPU. On TPU, torch_xla builds a lazy graph that
    only actually runs when something forces it - a host read
    (`.item()`/`.cpu()`) or an explicit mark. This project's pipeline
    loop already forces plenty of syncs itself (the last stage's
    `loss.item()` every step, every non-last rank's `.cpu()` activation/
    gradient send every step), so training is CORRECT without this -
    but leaving each step's graph to be closed off implicitly by
    whichever host read happens to come along, rather than as a real
    step boundary, was measurably slower in this project's own
    validation run (~2x) - so call this once per step, right after
    `optimizer.step()`, for real amortized compile/execute batching
    instead of relying on that side effect."""
    if device.type == "xla":
        import torch_xla

        torch_xla.sync()


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
