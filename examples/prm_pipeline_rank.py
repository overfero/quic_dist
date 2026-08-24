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

from quic_dist.cli import run_pipeline_rank_main
from quic_dist.rlhf import PRMConfig, run_prm_training

run_pipeline_rank_main(PRMConfig, run_prm_training, default_job_id="prm_pipeline")
