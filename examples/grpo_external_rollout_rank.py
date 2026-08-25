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


def _rollout_source():
    step = 1
    while True:
        pt_path = os.path.join(rollout_dir, f"step_{step:06d}.pt")
        while not os.path.exists(pt_path):
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


run_grpo_training_from_rollouts(
    rank, signaling_url, config, _rollout_source(), job_id=job_id,
    max_steps=max_steps, on_step_result=_write_result,
)
