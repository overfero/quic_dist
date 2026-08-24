"""Generic launcher for quic_dist.distill's config-driven knowledge
distillation. rank 0 = teacher, rank 1 = student - always world_size=2,
no rank/topology args needed beyond which of the two this process is.

Usage: python3 distill_pipeline_rank.py <config.yaml> <rank> <signaling_url> [job_id]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quic_dist.cli import run_pipeline_rank_main
from quic_dist.distill import DistillConfig, run_distill_training

run_pipeline_rank_main(DistillConfig, run_distill_training, default_job_id="distill_pipeline")
