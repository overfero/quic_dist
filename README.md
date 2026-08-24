# quic_dist

A real `torch.distributed.ProcessGroup`/`Store` backend over QUIC, registered
as `"quic"`. Built for pipeline-parallel training (LoRA, QLoRA, or full
fine-tuning) between machines that are behind NAT, with no cloud VPC, no
port forwarding, and no direct reachability — only that every rank can reach
one shared HTTP signaling server.

Standalone: this package has no dependency on vLLM or any other project.
It vendors its own copy of the UDP hole-punch client (`holepunch/peer.py`)
and its own small Rust workspace (`rust/`) for the QUIC engine.

## Install

```bash
git clone <this-repo-url> quic_dist   # directory name doesn't matter once installed
cd quic_dist
pip install -e .                      # installs the `quic_dist` package + Python deps

cd rust && cargo build --release      # builds the compiled QUIC engine
cp target/release/lib_rust_quic_engine.so ../_rust_quic_engine.abi3.so
cd ..
```

Requires a Rust toolchain (matching `rust/rust-toolchain.toml`, installed
automatically via [rustup](https://rustup.rs) if you don't already have one)
to build the QUIC engine. If you only need the signaling server, install
with `pip install -e ".[signaling]"` for its `fastapi`/`uvicorn` deps.

## Usage

```python
import quic_dist

quic_dist.init_process_group(
    signaling_url="https://your-signaling-server",
    rank=rank,
    world_size=world_size,
)

# from here on, standard torch.distributed - nothing quic_dist-specific:
import torch.distributed as dist
dist.send(tensor, dst=1, tag=7)
dist.recv(tensor, src=1, tag=7)
dist.barrier()
```

Run the signaling server (needed once, reachable by every rank):

```bash
cd holepunch
uvicorn signaling_server:app --host 0.0.0.0 --port 8000
```

## Scope

Point-to-point communication only: `send`/`recv`/`isend`/`irecv`/`barrier`.
Every collective (`all_reduce`, `all_gather`, `broadcast`, ...) raises
`NotImplementedError` by explicit design — not a silent no-op, not a
fallback to another backend.

## Config-driven pipeline LoRA/QLoRA fine-tuning

`quic_dist.finetune` extracts the boilerplate that's identical across any
N-stage pipeline-parallel LoRA/QLoRA run over this transport — per-rank
`device_map` construction (including uneven layer splits, for a boundary
stage carrying extra fixed weight like a large `lm_head`), the generic
recv→layers→send forward/backward loop for any world size, and the
quic_dist init/teardown. A new causal-LM model/dataset needs a config, not
a new training script:

```bash
pip install -e ".[finetune]"   # transformers, peft, bitsandbytes, accelerate, datasets, pyyaml
cd examples
python3 pipeline_finetune_rank.py configs/qwen25_0.5b_lora.yaml <rank> <signaling_url>
```

See `examples/configs/` for two real, validated examples — a 2-stage plain
LoRA run (`qwen25_0.5b_lora.yaml`) and a 4-stage real QLoRA run on a ~27B
hybrid-attention model across 2 real machines
(`qwen38_27b_qlora.yaml`, loss 0.6973 → 0.2734 over 3 real epochs) — and
`quic_dist/finetune.py`'s own docstring for what's deliberately NOT
covered (non-text modalities stay their own script; see
`vision_pipeline_rank.py`).

## Config-driven pipeline DPO/GRPO/PPO

`quic_dist.rlhf` covers RLHF-style training the same way `finetune.py`
covers SFT: a config, not a new script. All three reuse `finetune.py`'s
model-loading (`build_stage_model`) directly, and DPO/GRPO/PPO all get
their reference-model log-probs for free via peft's `disable_adapter()`
context manager on the SAME model - no second model is ever loaded.
GRPO and PPO add real pipeline-parallel autoregressive generation
(`rlhf.pipeline_generate` - each new token is one full round trip
through every stage, with each stage keeping its own local KV cache
across steps); PPO additionally attaches a real learned value head at
the last stage and runs genuine GAE + a clipped surrogate loss over
`ppo_epochs` inner update passes per rollout (GRPO deliberately skips
that inner-epoch loop - see `GRPOConfig`'s docstring for why its
importance ratio is always 1, unlike PPO's).

```bash
cd examples
python3 dpo_pipeline_rank.py configs/qwen25_0.5b_dpo.yaml <rank> <signaling_url>
python3 grpo_pipeline_rank.py configs/qwen25_0.5b_grpo.yaml <rank> <signaling_url>
python3 ppo_pipeline_rank.py configs/qwen25_0.5b_ppo.yaml <rank> <signaling_url>
```

All three configs are real, validated (loopback) examples on
Qwen2.5-0.5B: DPO loss 0.6931 → 0.3663 (avg last 8 steps), GRPO reward
0.775 → 0.803, PPO reward held ~0.72-0.75 while the clipped policy loss
and value loss both dropped. `rlhf.default_reward_fn` (GRPO/PPO) is a
real, deterministic, rule-based scorer (lexical diversity + a length
target) - documented in its own docstring as a stand-in good enough to
prove the mechanism, not a trained reward model; pass your own
`reward_fn` to `run_grpo_training`/`run_ppo_training` for a real one.
`rlhf.RMConfig`/`run_rm_training` and `rlhf.PRMConfig`/`run_prm_training`
train real Bradley-Terry outcome and per-step process reward models;
`load_reward_model`/`pipeline_score` reload one to score PPO/GRPO
rollouts with an actual trained model instead of `reward_fn`.
`finetune.PipelineConfig.training_mode` ("cpt", the original all-tokens
behavior, or "sft", response-only masked loss) covers continued
pretraining vs. instruction tuning through the same module.

## Config-driven pipeline multimodal (vision-language) SFT

`quic_dist.multimodal` extends the same config/dotted-attribute
philosophy to a real vision-language model: a LLaVA-family checkpoint
(SigLIP vision tower + a plain Qwen2 decoder - picked specifically to
avoid models needing special multimodal position encoding like Qwen2-VL's
M-RoPE, which would need per-image metadata broadcast to every stage,
not just a config field). Rank 0 runs the real vision tower + projector
(the model's own `get_image_features`, not reimplemented) and merges
image embeddings into the text embedding sequence via the exact
`masked_scatter` logic `LlavaModel.forward` itself uses; every later
stage sees a plain `(B, T, hidden)` activation, no changes needed.

```bash
cd examples
python3 multimodal_pipeline_rank.py configs/llava_qwen05b_multimodal.yaml <rank> <signaling_url>
```

Real, validated example on `llava-hf/llava-interleave-qwen-0.5b-hf`
(SigLIP + Qwen2-0.5B) with `HuggingFaceH4/llava-instruct-mix-vsft` (real
images, real chat-format VQA turns): loopback loss 2.44 → 1.73 (avg
first/last 8 of 48 steps, 3 epochs), and CONFIRMED CROSS-MACHINE (2 real
machines, one real network hop) with the same result - loss trajectory
matched the loopback run step-for-step, 206.8s total.

## Tests

```bash
cd tests
pip install pytest
python3 -m pytest test_store.py test_process_group.py test_pipeline.py test_parallel_stream.py -q
```

These spin up a real local signaling server and real hole-punched QUIC
connections (loopback) — no mocking, and live in `tests/` alongside
their own `_helpers.py`. `examples/` holds the standalone,
directly-run scripts (`cross_machine_rank.py`, `lora_pipeline_rank.py`,
`real_llm_pipeline_rank.py`, `vision_shape_rank.py`,
`vision_pipeline_rank.py`, `n3_cross_machine_rank.py`,
`pipeline_finetune_rank.py`, `dpo_pipeline_rank.py`,
`grpo_pipeline_rank.py`, `ppo_pipeline_rank.py`, `rm_pipeline_rank.py`,
`prm_pipeline_rank.py`, `multimodal_pipeline_rank.py`,
`bench_quic_dist.py`)
plus the `configs/` and `data/` they read from - run one instance per
machine/rank, pointed at a real publicly reachable signaling URL.

## Known limitations

See the full deliverable report for the complete architecture, design
rationale, and an honest list of what's implemented vs. not:
https://claude.ai/code/artifact/6677daa1-801f-44c9-b6b5-d42880632c6b
