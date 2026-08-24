"""Generic launcher for quic_dist.finetune's config-driven pipeline LoRA/
QLoRA training - the actual per-model scripts (real_llm_pipeline_rank.py,
qwen38_27b_pipeline_rank.py) are kept as-is since they were the real,
directly-validated proofs this module was extracted FROM, but any NEW
causal-LM model/dataset combination should go through this launcher plus
a config file, not a new copy of the training loop.

Usage: python3 pipeline_finetune_rank.py <config.yaml> <rank> <signaling_url> [job_id]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quic_dist.cli import run_pipeline_rank_main
from quic_dist.finetune import PipelineConfig, run_pipeline_training

run_pipeline_rank_main(PipelineConfig, run_pipeline_training, default_job_id="pipeline_finetune")
