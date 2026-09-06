"""Launcher for rlhf.run_grpo_training_from_rollouts, fed by a real
vLLM-TPU-generated rollout file (examples/vllm_generate_rollout.py,
run first, in vLLM's own venv/process - see that script's own module
docstring for why this is a SEQUENTIAL handoff, not concurrent
chip-sharing with this process).

Usage:
  python3 grpo_math_from_vllm_rollout.py <config.yaml> <rank> <signaling_url> <rollout.json> [job_id] [wandb_project] [wandb_run_name]

WANDB_API_KEY must be set in the environment when wandb_project is
given - never accepted as a CLI arg/config field.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quic_dist.rlhf import GRPOConfig, load_rollouts_from_vllm_json, run_grpo_training_from_rollouts

config_path = sys.argv[1]
rank = int(sys.argv[2])
signaling_url = sys.argv[3]
rollout_path = sys.argv[4]
job_id = sys.argv[5] if len(sys.argv) > 5 else "grpo_math_vllm_pipeline"
wandb_project = sys.argv[6] if len(sys.argv) > 6 else None
wandb_run_name = sys.argv[7] if len(sys.argv) > 7 else None

config = GRPOConfig.from_file(config_path)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(config.model_path)
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
rollouts = load_rollouts_from_vllm_json(rollout_path, pad_token_id)
print(f"[rank {rank}] loaded {len(rollouts)} rollout batches from {rollout_path}", flush=True)

wb = None
if wandb_project is not None and rank == config.world_size - 1:
    import os
    import wandb

    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        raise ValueError("wandb_project given but WANDB_API_KEY is not set in the environment.")
    wandb.login(key=api_key)
    wb = wandb.init(project=wandb_project, name=wandb_run_name, config={
        k: v for k, v in vars(config).items() if isinstance(v, (int, float, str, bool, type(None)))
    })


def on_step_result(step_counter, loss_value, reward_mean, kl_value):
    if wb is not None and loss_value is not None:
        wb.log({
            "train/loss": loss_value, "train/reward_mean": reward_mean,
            "train/accuracy": reward_mean, "train/kl": kl_value if kl_value is not None else 0.0,
        }, step=step_counter)


losses = run_grpo_training_from_rollouts(
    rank, signaling_url, config, rollouts, job_id=job_id,
    on_step_result=on_step_result if wb is not None else None,
)

if wb is not None:
    wb.finish()
print(f"[rank {rank}] DONE, losses={losses}", flush=True)
