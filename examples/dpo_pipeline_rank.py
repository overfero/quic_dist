"""Generic launcher for quic_dist.rlhf's config-driven pipeline DPO.

Usage: python3 dpo_pipeline_rank.py <config.yaml> <rank> <signaling_url> [job_id]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quic_dist.cli import run_pipeline_rank_main
from quic_dist.rlhf import DPOConfig, run_dpo_training

run_pipeline_rank_main(DPOConfig, run_dpo_training, default_job_id="dpo_pipeline")
