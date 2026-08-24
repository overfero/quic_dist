"""Generic launcher for quic_dist.multimodal's config-driven pipeline
vision-language SFT.

Usage: python3 multimodal_pipeline_rank.py <config.yaml> <rank> <signaling_url> [job_id]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quic_dist.cli import run_pipeline_rank_main
from quic_dist.multimodal import MultimodalConfig, run_multimodal_training

run_pipeline_rank_main(MultimodalConfig, run_multimodal_training, default_job_id="multimodal_pipeline")
