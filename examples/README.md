# Examples

**Before running any script here, start the signaling server once** —
every rank (on every machine involved) needs to reach it:

```bash
cd ../holepunch
uvicorn signaling_server:app --host 0.0.0.0 --port 8000
```

Then run one instance of the relevant script **per rank**, each pointed
at that same `signaling_url` — on the same machine (loopback) or on
genuinely different machines (real cross-machine, real hole-punch), your
choice; nothing about the scripts changes either way.

## Two families

**Config-driven** (`<config.yaml> <rank> <signaling_url> [job_id]`) — a
config file selects the model/dataset/hyperparameters, the script itself
never changes. Add a new training run by writing a new YAML under
`configs/`, not a new script. Since this session's polish pass, 8 of
these collapse to a one-line call to `quic_dist.cli.run_pipeline_rank_main`
(`cli.py` at the repo root); `grpo_pipeline_rank.py`/`ppo_pipeline_rank.py`
have real extra logic (optional reward-model loading) so they still parse
`sys.argv` directly.

**Hand-rolled** (rank-first, no config file) — pedagogical or
proof-of-concept scripts, each with its own model/data baked in.
**`lora_pipeline_rank.py` is the one to watch out for**: its filename
matches the config-driven family, but it's actually hand-rolled — a
from-scratch `nn.TransformerEncoderLayer` toy model, no config argument.

## All scripts

| Script | Family | Usage | Needs |
|---|---|---|---|
| `pipeline_finetune_rank.py` | config-driven | `<config.yaml> <rank> <signaling_url> [job_id]` | `quic_dist.finetune.PipelineConfig` — LoRA/QLoRA SFT/CPT, any causal-LM |
| `dpo_pipeline_rank.py` | config-driven | `<config.yaml> <rank> <signaling_url> [job_id]` | `quic_dist.rlhf.DPOConfig` |
| `rloo_pipeline_rank.py` | config-driven | `<config.yaml> <rank> <signaling_url> [job_id]` | `quic_dist.rlhf.RLOOConfig` |
| `rm_pipeline_rank.py` | config-driven | `<config.yaml> <rank> <signaling_url> [job_id]` | `quic_dist.rlhf.RMConfig` — reward model, used by PPO/GRPO's optional real-RM mode |
| `prm_pipeline_rank.py` | config-driven | `<config.yaml> <rank> <signaling_url> [job_id]` | `quic_dist.rlhf.PRMConfig` — process reward model |
| `distill_pipeline_rank.py` | config-driven | `<config.yaml> <rank> <signaling_url> [job_id]` | `quic_dist.distill.DistillConfig` — rank 0 = teacher, rank 1 = student, always world_size=2 |
| `pretrain_pipeline_rank.py` | config-driven | `<config.yaml> <rank> <signaling_url> [job_id]` | `quic_dist.pretrain.PretrainConfig` — from-scratch, no LoRA/quant |
| `multimodal_pipeline_rank.py` | config-driven | `<config.yaml> <rank> <signaling_url> [job_id]` | `quic_dist.multimodal.MultimodalConfig` — vision-language SFT |
| `grpo_pipeline_rank.py` | config-driven (extra args) | `<config.yaml> <rank> <signaling_url> [job_id] [rm_config.yaml] [rm_checkpoint_dir]` | `quic_dist.rlhf.GRPOConfig`; the two trailing args are optional — omit for the built-in rule-based reward, pass both for a real trained RM |
| `ppo_pipeline_rank.py` | config-driven (extra args) | `<config.yaml> <rank> <signaling_url> [job_id] [rm_config.yaml] [rm_checkpoint_dir]` | `quic_dist.rlhf.PPOConfig`; same optional real-RM args as GRPO above |
| `cross_machine_rank.py` | hand-rolled | `<rank> <signaling_url> [job_id]` | nothing — a 2-rank send/recv/isend/irecv/barrier + 10MB throughput correctness check, the first real cross-machine proof |
| `n3_cross_machine_rank.py` | hand-rolled | `<rank> <signaling_url> [job_id]` | nothing — 3 genuinely separate machines, rank 1 holds two concurrent cross-machine connections at once |
| `lora_pipeline_rank.py` | hand-rolled (misleadingly named) | `<rank> <signaling_url> [job_id]` | nothing — hand-rolled `nn.TransformerEncoderLayer` toy model, **no config file**, despite the filename |
| `vision_shape_rank.py` | hand-rolled | `<rank> <signaling_url> [job_id]` | nothing — proves `tensor.py` is genuinely shape-agnostic (4D/5D tensors), not a training run |
| `vision_pipeline_rank.py` | hand-rolled | `<rank> <signaling_url> [job_id]` | downloads `WinKawaks/vit-tiny-patch16-224` — real pretrained ViT LoRA fine-tuning |
| `real_llm_pipeline_rank.py` | hand-rolled (extra flag) | `<rank> <signaling_url> [--mode lora\|qlora] [job_id]` | downloads `Qwen/Qwen2.5-0.5B` |
| `qwen38_27b_pipeline_rank.py` | hand-rolled | `<rank> <signaling_url> [job_id]` | downloads `Qwen/Qwen3.8-27B`, 4 stages, needs 4 real GPUs across 2 machines |
| `bench_quic_dist.py` | standalone | `python3 bench_quic_dist.py` (no args — spawns its own signaling server + 2 loopback processes) | nothing — see `docs/profiling.md` for reading its output alongside `QUIC_DIST_RUST_DEBUG=1` |

`configs/` holds every config-driven script's YAML files; `data/` holds
`tinyshakespeare.txt`, the fixed small dataset a couple of the
hand-rolled scripts read directly.

## Full walkthrough (config-driven LoRA SFT, 2 ranks, loopback)

```bash
# terminal 1: signaling server (leave running)
cd holepunch && uvicorn signaling_server:app --host 0.0.0.0 --port 8000

# terminal 2: rank 0
pip install -e ".[finetune]"   # transformers, peft, bitsandbytes, accelerate, datasets, pyyaml (from the repo root)
cd examples
python3 pipeline_finetune_rank.py configs/qwen25_0.5b_lora.yaml 0 http://127.0.0.1:8000

# terminal 3: rank 1
cd examples
python3 pipeline_finetune_rank.py configs/qwen25_0.5b_lora.yaml 1 http://127.0.0.1:8000
```

Both ranks connect to the signaling server, hole-punch to each other
(instant on loopback), split `Qwen/Qwen2.5-0.5B`'s decoder layers across
the 2 ranks per the config's `stage_layer_counts`, and run real
pipeline-parallel LoRA SFT — rank 1 (the last stage) prints the loss
each step. Point `signaling_url` at a real publicly reachable server and
run each rank on a genuinely different machine for a real cross-machine
run — nothing else changes.
