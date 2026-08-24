"""Shared launcher boilerplate for the config-driven `examples/
*_pipeline_rank.py` scripts (`<config.yaml> <rank> <signaling_url>
[job_id]`) - extracted because 8 of them hand-copied this exact
~10-line argv-parsing+bootstrap shape identically. Scripts with real
extra logic beyond this shape (`grpo_pipeline_rank.py`/
`ppo_pipeline_rank.py`'s optional reward-model args) are NOT forced
through this helper - they parse `sys.argv` themselves.

Not part of the transport or training-loop code - this is pure CLI
glue, imported only by `examples/*_pipeline_rank.py` scripts, never by
`process_group.py`/`finetune.py`/`rlhf.py`/etc. themselves."""
from __future__ import annotations

import sys
from typing import Callable


def run_pipeline_rank_main(config_cls, run_fn: Callable, default_job_id: str) -> None:
    """Parses `sys.argv` as `<config.yaml> <rank> <signaling_url>
    [job_id]`, builds `config_cls.from_file(config_path)`, and calls
    `run_fn(rank, signaling_url, config, job_id=job_id)` - the exact
    shape every config-driven rank script in `examples/` already used
    by hand. `job_id` defaults to `default_job_id` when not passed on
    the command line, matching each script's own previous default."""
    config_path = sys.argv[1]
    rank = int(sys.argv[2])
    signaling_url = sys.argv[3]
    job_id = sys.argv[4] if len(sys.argv) > 4 else default_job_id

    config = config_cls.from_file(config_path)
    run_fn(rank, signaling_url, config, job_id=job_id)
