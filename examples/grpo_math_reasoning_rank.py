"""Launcher for rlhf.run_grpo_math_training - a real GSM8K/MATH-style
math-reasoning GRPO recreation (see that function's own docstring for
why this is a separate training loop from run_grpo_training, not a
config variant of it).

Not run_pipeline_rank_main-based (see cli.py's own module docstring
for why: scripts needing arguments beyond that helper's fixed
<config> <rank> <signaling_url> [job_id] shape parse sys.argv
themselves - this one needs optional wandb_project/wandb_run_name).

Usage:
  python3 grpo_math_reasoning_rank.py <config.yaml> <rank> <signaling_url> [job_id] [wandb_project] [wandb_run_name]

WANDB_API_KEY must be set in the environment when wandb_project is
given - never accepted as a CLI arg/config field, so it can't end up
in a config file or a shell history search for one.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quic_dist.rlhf import GRPOConfig, run_grpo_math_training

config_path = sys.argv[1]
rank = int(sys.argv[2])
signaling_url = sys.argv[3]
job_id = sys.argv[4] if len(sys.argv) > 4 else "grpo_math_pipeline"
wandb_project = sys.argv[5] if len(sys.argv) > 5 else None
wandb_run_name = sys.argv[6] if len(sys.argv) > 6 else None

config = GRPOConfig.from_file(config_path)
run_grpo_math_training(rank, signaling_url, config, job_id=job_id,
                        wandb_project=wandb_project, wandb_run_name=wandb_run_name)
