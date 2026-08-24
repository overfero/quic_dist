"""Generic launcher for quic_dist.rlhf's config-driven pipeline RLOO/REINFORCE.

Usage: python3 rloo_pipeline_rank.py <config.yaml> <rank> <signaling_url> [job_id]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quic_dist.rlhf import RLOOConfig, run_rloo_training

config_path = sys.argv[1]
rank = int(sys.argv[2])
signaling_url = sys.argv[3]
job_id = sys.argv[4] if len(sys.argv) > 4 else "rloo_pipeline"

config = RLOOConfig.from_file(config_path)
run_rloo_training(rank, signaling_url, config, job_id=job_id)
