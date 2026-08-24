"""Generic launcher for quic_dist.rlhf's config-driven pipeline reward
model (RM) training - Bradley-Terry pairwise loss on preference pairs,
producing a checkpoint (LoRA adapter + reward_head per rank, see
rlhf.save_reward_stage()) usable as PPO/GRPO's real reward signal via
ppo_pipeline_rank.py's/grpo_pipeline_rank.py's optional RM args.

Usage: python3 rm_pipeline_rank.py <config.yaml> <rank> <signaling_url> [job_id]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quic_dist.cli import run_pipeline_rank_main
from quic_dist.rlhf import RMConfig, run_rm_training

run_pipeline_rank_main(RMConfig, run_rm_training, default_job_id="rm_pipeline")
