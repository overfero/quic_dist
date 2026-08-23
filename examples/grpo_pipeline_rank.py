"""Generic launcher for quic_dist.rlhf's config-driven pipeline GRPO.

Optionally scores rollouts with a REAL trained reward model instead of
the default rule-based reward_fn - see ppo_pipeline_rank.py's identical
extra-args convention (this is the same rm= wiring, just for GRPO).

Usage: python3 grpo_pipeline_rank.py <config.yaml> <rank> <signaling_url> [job_id] [rm_config.yaml] [rm_checkpoint_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quic_dist.rlhf import GRPOConfig, RMConfig, run_grpo_training, load_reward_model

config_path = sys.argv[1]
rank = int(sys.argv[2])
signaling_url = sys.argv[3]
job_id = sys.argv[4] if len(sys.argv) > 4 else "grpo_pipeline"
rm_config_path = sys.argv[5] if len(sys.argv) > 5 else None
rm_checkpoint_dir = sys.argv[6] if len(sys.argv) > 6 else None

config = GRPOConfig.from_file(config_path)

rm = None
if rm_config_path and rm_checkpoint_dir:
    import torch

    rm_config = RMConfig.from_file(rm_config_path)
    local_gpu = rank % torch.cuda.device_count()
    rm = load_reward_model(rank, local_gpu, rm_config, rm_checkpoint_dir)

run_grpo_training(rank, signaling_url, config, job_id=job_id, rm=rm)
