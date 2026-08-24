"""Generic launcher for quic_dist.pretrain's config-driven pipeline
pretraining-from-scratch.

Usage: python3 pretrain_pipeline_rank.py <config.yaml> <rank> <signaling_url> [job_id]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quic_dist.cli import run_pipeline_rank_main
from quic_dist.pretrain import PretrainConfig, run_pretrain_training

run_pipeline_rank_main(PretrainConfig, run_pretrain_training, default_job_id="pretrain_pipeline")
