"""Launcher for quic_dist.rlhf's run_grpo_training_from_rollouts - reads
RolloutBatch items from files quic-rl's QuicTrainBackend writes to a
shared directory, instead of generating rollouts in-process. Same GRPO
update math as grpo_pipeline_rank.py (both call into
rlhf._grpo_update_from_rollout); only the rollout source differs - see
run_grpo_training_from_rollouts's own docstring.

File protocol (quic-rl's own convention, not part of quic_dist's public
API): the writer (QuicTrainBackend.train()) atomically writes
torch.save()'d RolloutBatch objects to `{rollout_dir}/step_{N:06d}.pt`
in strictly increasing N starting at 1 (write to a `.tmp` sibling then
os.rename() - atomic on the same filesystem, so a poller here never
observes a partially-written file). This script polls for each N in
order and yields it as soon as it appears. The LAST rank additionally
writes `{rollout_dir}/step_{N:06d}.result.json` right after finishing
step N (via run_grpo_training_from_rollouts's on_step_result hook), so
QuicTrainBackend can read back the real loss/reward/kl for that step
without any process communication beyond the filesystem both sides
already share for the rollout handoff itself.

Usage: python3 grpo_external_rollout_rank.py <config.yaml> <rank> <signaling_url> <rollout_dir> [job_id] [max_steps]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
# rlhf.py itself does bare `from training_utils import ...` (a real,
# pre-existing pattern shared by every RLHF mode, not something to
# change here) - that only resolves when quic_dist's OWN repo root is
# directly on sys.path, not just its parent (which is enough for
# `import quic_dist.rlhf` alone). A real bug hit running this for real:
# checkpoint_dir was never set in any earlier validation, so this
# import path was never actually exercised until now.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from quic_dist.rlhf import GRPOConfig, run_grpo_training_from_rollouts

if os.environ.get("QUIC_DIST_FAULTHANDLER"):
    # Matches n3_cross_machine_rank.py's own use of this - py-spy needs
    # ptrace, which this container's seccomp profile blocks outright;
    # faulthandler dumps every thread's real Python stack from IN-PROCESS
    # via a plain signal, no ptrace needed. Added here specifically to
    # debug a real, reproducible hang found running this for real: both
    # ranks reach "starting training"/hole-punch success, QUIC-level
    # stats show a healthy idle connection (no loss, no blocking frames,
    # cwnd fine) with zero new packets sent for the rest of the run - so
    # the hang is in Python/application logic, not the transport, and
    # this is how to find out exactly which line.
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)

config_path = sys.argv[1]
rank = int(sys.argv[2])
signaling_url = sys.argv[3]
rollout_dir = sys.argv[4]
job_id = sys.argv[5] if len(sys.argv) > 5 else "grpo_pipeline_external"
max_steps = int(sys.argv[6]) if len(sys.argv) > 6 else None

config = GRPOConfig.from_file(config_path)
is_last = rank == config.world_size - 1

_POLL_INTERVAL_S = 0.5

# Populated by _on_checkpoint() the first time it fires (i.e. once a real
# training step has actually happened) - read by _rollout_source()'s own
# idle-poll loop below. Plain module-level dict, not a local var, since
# it needs to be visible to a generator defined here AND updated from a
# callback invoked from inside run_grpo_training_from_rollouts's own
# frame - a real deadlock was found without this (see _rollout_source's
# own comment on the fix).
_state = {"peft_model": None}


def _rollout_source():
    step = 1
    while True:
        pt_path = os.path.join(rollout_dir, f"step_{step:06d}.pt")
        while not os.path.exists(pt_path):
            # Real deadlock found running this for real: export_policy()
            # is normally called right after train() returns, i.e.
            # exactly while this rank is sitting HERE waiting for the
            # NEXT batch (which the orchestrator won't write until AFTER
            # export_policy() itself returns) - on_checkpoint alone only
            # fires from inside an active training step, so an export
            # request arriving during this idle wait would never get
            # serviced, and export_policy() would block forever waiting
            # on a rank that's waiting right back on it. Checking here
            # too, using whatever model _on_checkpoint last stashed,
            # closes that window.
            _check_export_request(rank, _state["peft_model"])
            time.sleep(_POLL_INTERVAL_S)
        # weights_only=False: RolloutBatch is our own dataclass, not a
        # state_dict of tensors alone - torch's default-safe loader
        # rejects arbitrary classes.
        batch = torch.load(pt_path, weights_only=False)
        yield batch
        step += 1


def _write_result(step: int, loss_value, reward_mean, kl_value) -> None:
    if loss_value is None:
        return  # non-last rank; nothing to report
    result_path = os.path.join(rollout_dir, f"step_{step:06d}.result.json")
    tmp_path = result_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({"loss": loss_value, "reward_mean": reward_mean, "kl": kl_value}, f)
    os.rename(tmp_path, result_path)  # atomic - QuicTrainBackend never sees a partial file


def _write_done_marker(request_id, output_dir: str) -> None:
    done_path = os.path.join(rollout_dir, f"export_done_{request_id}.json")
    tmp_path = done_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({"output_dir": output_dir}, f)
    os.rename(tmp_path, done_path)


def _export_via_shard_merge(this_rank: int, peft_model, output_dir: str, request_id, request_path: str) -> None:
    """full_finetune=True + world_size>1 export path: since
    build_stage_model()'s own memory fix means NO SINGLE RANK holds a
    complete model any more (each rank only keeps its own owned layers
    plus whichever of embed_tokens/norm/lm_head it owns - a real,
    necessary consequence of fixing a real OOM, not a regression to work
    around quietly), every rank writes its OWN shard here, and rank 0
    additionally coordinates: waits for every rank's shard to land, then
    calls finetune.merge_stage_shards() to reassemble one complete model
    and save it. `resolve_attr` against THIS rank's own already-trimmed
    `peft_model` (not a param passed through on_checkpoint's contract,
    which stays unchanged) is what's actually owned right now - correct
    by construction, no coordination needed to know what's ours."""
    from quic_dist.finetune import merge_stage_shards, resolve_attr, stage_range

    my_range = stage_range(this_rank, config)
    layers = resolve_attr(peft_model, config.layers_attr)
    embed = resolve_attr(peft_model, config.embed_attr)
    norm = resolve_attr(peft_model, config.norm_attr)
    lm_head = resolve_attr(peft_model, config.lm_head_attr)

    shard_path = os.path.join(rollout_dir, f"export_shard_rank{this_rank}_{request_id}.pt")
    if not os.path.exists(shard_path):
        shard = {
            "layer_start": my_range.start,
            "layer_state_dicts": [layer.state_dict() for layer in layers],
            "embed_state_dict": embed.state_dict() if embed is not None else None,
            "norm_state_dict": norm.state_dict() if norm is not None else None,
            "lm_head_state_dict": lm_head.state_dict() if lm_head is not None else None,
        }
        tmp_path = shard_path + ".tmp"
        torch.save(shard, tmp_path)
        os.rename(tmp_path, shard_path)

    if this_rank != 0:
        return  # only rank 0 coordinates the merge below

    shard_paths = [os.path.join(rollout_dir, f"export_shard_rank{r}_{request_id}.pt") for r in range(config.world_size)]
    if not all(os.path.exists(p) for p in shard_paths):
        return  # not every rank has written its shard yet - retry on the next poll tick

    os.makedirs(output_dir, exist_ok=True)
    merge_stage_shards(config.model_path, shard_paths, config, output_dir)
    os.remove(request_path)  # consume it - never re-triggers on the next check
    for p in shard_paths:
        os.remove(p)
    _write_done_marker(request_id, output_dir)


def _check_export_request(this_rank: int, peft_model) -> None:
    """Services QuicTrainBackend.export_policy()'s file-based request.
    Called from two places (see each call site's own comment for why
    both are needed): _on_checkpoint (right after a real training step)
    and _rollout_source's own idle-poll loop (while waiting for the
    NEXT step - closes a real deadlock the first call site alone can't).
    Serializes DIRECTLY from this already-GPU-resident `peft_model` -
    the whole reason this hook exists rather than letting the
    orchestrator load a second full model copy into its own process (a
    real RAM-exhaustion bug hit running this for real: rank processes
    never exit between train() calls, and a second full-model load in a
    separate orchestrator process while both stayed resident pushed a
    real machine over its actual RAM ceiling).

    full_finetune=True + world_size>1 goes through the shard-then-merge
    path (every rank participates - see _export_via_shard_merge); every
    other combination (LoRA/quantized, or world_size==1) still has a
    single rank holding a genuinely complete model (LoRA's device_map
    already keeps every layer as a real module, just some offloaded -
    see build_device_map's own docstring), so rank 0 alone
    save_pretrained()-ing is still correct and cheaper there."""
    if peft_model is None:
        return
    request_path = os.path.join(rollout_dir, "export_request.json")
    if not os.path.exists(request_path):
        return
    with open(request_path) as f:
        req = json.load(f)
    output_dir = req["output_dir"]
    request_id = req["request_id"]

    if config.full_finetune and config.world_size > 1:
        _export_via_shard_merge(this_rank, peft_model, output_dir, request_id, request_path)
        return

    if this_rank != 0:
        return
    os.makedirs(output_dir, exist_ok=True)
    peft_model.save_pretrained(output_dir, safe_serialization=True)
    os.remove(request_path)  # consume it - never re-triggers on the next check
    _write_done_marker(request_id, output_dir)


def _on_checkpoint(this_rank: int, peft_model) -> None:
    _state["peft_model"] = peft_model
    _check_export_request(this_rank, peft_model)


run_grpo_training_from_rollouts(
    rank, signaling_url, config, _rollout_source(), job_id=job_id,
    max_steps=max_steps, on_step_result=_write_result, on_checkpoint=_on_checkpoint,
)
