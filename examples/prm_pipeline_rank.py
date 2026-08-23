"""Generic launcher for quic_dist.rlhf's config-driven pipeline process
reward model (PRM) training - per-reasoning-step binary classification
on peiyi9979/Math-Shepherd, producing a checkpoint (LoRA adapter +
reward_head per rank, see rlhf.save_reward_stage()) usable for
best-of-N reranking via rlhf.pipeline_score_prm().

Usage: python3 prm_pipeline_rank.py <config.yaml> <rank> <signaling_url> [job_id]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quic_dist.rlhf import PRMConfig, run_prm_training

config_path = sys.argv[1]
rank = int(sys.argv[2])
signaling_url = sys.argv[3]
job_id = sys.argv[4] if len(sys.argv) > 4 else "prm_pipeline"

config = PRMConfig.from_file(config_path)
run_prm_training(rank, signaling_url, config, job_id=job_id)
