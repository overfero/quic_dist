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


def _check_export_request(this_rank: int, peft_model) -> None:
    """Services QuicTrainBackend.export_policy()'s file-based request.
    Called from two places (see each call site's own comment for why
    both are needed): _on_checkpoint (right after a real training step)
    and _rollout_source's own idle-poll loop (while waiting for the
    NEXT step - closes a real deadlock the first call site alone can't).
    Only rank 0 acts - every rank holds the full model in full_finetune
    mode (no per-rank sharding today - see build_stage_model's own
    docstring), so rank 0's copy alone already has everything, same
    reasoning export_policy() already uses reading checkpoint files.
    Serializes DIRECTLY from this already-GPU-resident `peft_model` -
    the whole reason this hook exists rather than letting the
    orchestrator load a second full model copy into its own process (a
    real RAM-exhaustion bug hit running this for real: rank processes
    never exit between train() calls, and a second full-model load in a
    separate orchestrator process while both stayed resident pushed a
    real machine over its actual RAM ceiling)."""
    if this_rank != 0 or peft_model is None:
        return
    request_path = os.path.join(rollout_dir, "export_request.json")
    if not os.path.exists(request_path):
        return
    with open(request_path) as f:
        req = json.load(f)
    output_dir = req["output_dir"]
    request_id = req["request_id"]
    os.makedirs(output_dir, exist_ok=True)
    peft_model.save_pretrained(output_dir, safe_serialization=True)
    os.remove(request_path)  # consume it - never re-triggers on the next check
    done_path = os.path.join(rollout_dir, f"export_done_{request_id}.json")
    tmp_path = done_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump({"output_dir": output_dir}, f)
    os.rename(tmp_path, done_path)


def _on_checkpoint(this_rank: int, peft_model) -> None:
    _state["peft_model"] = peft_model
    _check_export_request(this_rank, peft_model)


run_grpo_training_from_rollouts(
    rank, signaling_url, config, _rollout_source(), job_id=job_id,
    max_steps=max_steps, on_step_result=_write_result, on_checkpoint=_on_checkpoint,
)
