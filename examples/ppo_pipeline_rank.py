"""Generic launcher for quic_dist.rlhf's config-driven pipeline PPO.

Optionally scores rollouts with a REAL trained reward model (see
rm_pipeline_rank.py) instead of the default rule-based reward_fn - pass
an RM config + checkpoint dir as two extra args to make this genuine
end-to-end RLHF (SFT -> RM -> PPO) rather than the rule-based-reward
mechanism proof the bare 3-arg form gives you. The RM's world_size/rank
layout must match this PPO run's (see rlhf.load_reward_model's
docstring).

Usage: python3 ppo_pipeline_rank.py <config.yaml> <rank> <signaling_url> [job_id] [rm_config.yaml] [rm_checkpoint_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quic_dist.rlhf import PPOConfig, RMConfig, run_ppo_training, load_reward_model

config_path = sys.argv[1]
rank = int(sys.argv[2])
signaling_url = sys.argv[3]
job_id = sys.argv[4] if len(sys.argv) > 4 else "ppo_pipeline"
rm_config_path = sys.argv[5] if len(sys.argv) > 5 else None
rm_checkpoint_dir = sys.argv[6] if len(sys.argv) > 6 else None

config = PPOConfig.from_file(config_path)

rm = None
if rm_config_path and rm_checkpoint_dir:
    import torch

    rm_config = RMConfig.from_file(rm_config_path)
    local_gpu = rank % torch.cuda.device_count()
    rm = load_reward_model(rank, local_gpu, rm_config, rm_checkpoint_dir)

run_ppo_training(rank, signaling_url, config, job_id=job_id, rm=rm)
