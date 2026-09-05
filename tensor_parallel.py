"""Real intra-stage tensor parallelism (TP) via torch_xla's SPMD (GSPMD)
compiler - layered UNDER quic_dist's own pipeline parallelism (PP), not
a replacement for it. See PipelineConfig.tensor_parallel_size's own
docstring in finetune.py for how the two combine (a quic_dist PP rank
owns a contiguous BLOCK of `tensor_parallel_size` TPU chips instead of
one; TP shards each rank's own layers across that block).

Why this is possible at all despite `ProcessGroupQUIC` having NO
collective ops (see process_group.py's module docstring - all_reduce/
all_gather/broadcast raise NotImplementedError by explicit design): TP's
collectives never go through quic_dist's transport. SPMD is a
single-program-multi-data model - every op in this file (and every
Linear layer's own forward pass elsewhere in this codebase) is written
EXACTLY as it would be for one device; torch_xla's sharding propagation
inserts the real all-reduce/all-gather ENTIRELY INSIDE the compiled XLA
graph, between chips of the SAME rank's own local TP block. quic_dist's
transport only ever carries activations/gradients BETWEEN pipeline
stages (different ranks, potentially different machines) - a
fundamentally different communication axis that this module never
touches.

Megatron-style column/row sharding convention, but unlike Megatron's
original manual-collective implementation (separate forward code paths
issuing explicit collective calls), nothing here issues a collective -
this module ONLY calls `mark_sharding()` on WEIGHT tensors, never
touches activations, and never calls a collective op directly. Column-
parallel (q_proj/k_proj/v_proj/gate_proj/up_proj): shards a Linear's
OUTPUT features (weight dim 0) - safe to chain several back to back
since nothing needs the full output until a row-parallel layer
downstream consumes it. Row-parallel (o_proj/down_proj): shards INPUT
features (weight dim 1) - the layer whose raw per-chip output is a
PARTIAL SUM, made whole again by GSPMD's automatically-inserted
all-reduce (not written here - the compiler infers it from the
sharding annotation plus the matmul's contraction-dimension shape).

NOT sharded here, deliberately, as a real v1 scope cut: embed_tokens/
lm_head (vocab-parallel cross-entropy is a genuinely separate, more
involved piece - Megatron's own vocab-parallel loss - not implemented;
these stay replicated across the TP block, so this doesn't save memory
on a large vocab, only on the decoder layers' own linear weights) and
peft's own small lora_A/lora_B adapter matrices (r is tiny - 8 in every
config this repo validates - not worth sharding; they stay replicated,
and GSPMD's sharding propagation handles a sharded base_layer output
plus a replicated LoRA delta addition correctly without any special
handling here).
"""
from __future__ import annotations

import torch.nn as nn


# See this module's docstring for the column/row-parallel convention.
_COLUMN_PARALLEL_SUFFIXES = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "qkv_proj", "gate_up_proj")
_ROW_PARALLEL_SUFFIXES = ("o_proj", "down_proj")


def build_tp_mesh(tensor_parallel_size: int):
    """Builds a 1D SPMD Mesh, axis name "tensor", over this process's
    `tensor_parallel_size` locally-addressable TPU chips.
    training_utils.resolve_device() must have been called first with the
    SAME tensor_parallel_size (it's what restricts this process to
    exactly that many chips via TPU_VISIBLE_CHIPS and calls
    xr.use_spmd()) - a mismatch here is a real, immediate error from
    this function, not a silent partial mesh that would quietly shard
    across the wrong chip count."""
    import numpy as np
    import torch_xla.runtime as xr
    from torch_xla.distributed.spmd import Mesh

    n = xr.addressable_runtime_device_count()
    if n != tensor_parallel_size:
        raise ValueError(
            f"build_tp_mesh: {n} TPU chips addressable by this process, expected "
            f"tensor_parallel_size={tensor_parallel_size} - call "
            f"training_utils.resolve_device(rank, tensor_parallel_size=...) with the "
            f"same value first, before this function."
        )
    device_ids = np.array(range(n))
    return Mesh(device_ids, (n,), ("tensor",))


def _classify(leaf_name: str) -> str | None:
    if leaf_name in _COLUMN_PARALLEL_SUFFIXES:
        return "column"
    if leaf_name in _ROW_PARALLEL_SUFFIXES:
        return "row"
    return None


def _real_linear(sub: nn.Module) -> nn.Linear | None:
    """Returns the actual nn.Linear whose weight should be sharded - the
    module itself when it's a plain nn.Linear (full_finetune=True's
    path), or its `.base_layer` when it's a peft LoRA-wrapped layer
    (peft.tuners.lora.Linear wraps rather than subclasses nn.Linear, so
    a bare isinstance(sub, nn.Linear) check silently misses every
    LoRA-adapted q/k/v/o/gate/up/down_proj - this is what makes TP work
    under plain LoRA too, not just full_finetune)."""
    if isinstance(sub, nn.Linear):
        return sub
    base = getattr(sub, "base_layer", None)
    if isinstance(base, nn.Linear):
        return base
    return None


def shard_linear_layers(stage_layers, mesh) -> int:
    """Walks every decoder layer in `stage_layers` (a list/ModuleList of
    this quic_dist rank's OWN owned layers - never the whole model,
    matching PP's existing per-rank ownership) and mark_shardings each
    recognized attention/MLP Linear's weight (+ bias, column-parallel
    only - see this module's docstring for why row-parallel bias stays
    replicated). Returns how many Linear layers were sharded, so callers
    can assert it matches what they expected: a silently-unsharded run
    is a real, easy-to-miss memory/scaling regression (every chip still
    holds the FULL layer), not just a missed optimization - see
    finetune.py's run_pipeline_training for the assertion this guards."""
    import torch_xla.distributed.spmd as xs

    n_sharded = 0
    for layer in stage_layers:
        for name, sub in layer.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            kind = _classify(leaf)
            if kind is None:
                continue
            real = _real_linear(sub)
            if real is None:
                continue
            if kind == "column":
                xs.mark_sharding(real.weight, mesh, ("tensor", None))
                if real.bias is not None:
                    xs.mark_sharding(real.bias, mesh, ("tensor",))
            else:
                xs.mark_sharding(real.weight, mesh, (None, "tensor"))
                # bias intentionally left unsharded/replicated - it applies
                # to this layer's already-all-reduced (hence replicated)
                # output, see this module's docstring.
            n_sharded += 1
    return n_sharded
