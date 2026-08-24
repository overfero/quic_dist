# quic_dist

A real `torch.distributed.ProcessGroup`/`Store` backend over QUIC, registered
as `"quic"`. Built for pipeline-parallel training (LoRA, QLoRA, or full
fine-tuning) between machines that are behind NAT, with no cloud VPC, no
port forwarding, and no direct reachability — only that every rank can reach
one shared HTTP signaling server.

Standalone: this package has no dependency on vLLM or any other project.
It vendors its own copy of the UDP hole-punch client (`holepunch/peer.py`)
and its own small Rust workspace (`rust/`) for the QUIC engine.

## Contents

- [Install](#install)
- [Usage](#usage)
- [Scope](#scope)
- [Architecture](#architecture)
- [Config-driven pipeline LoRA/QLoRA fine-tuning](#config-driven-pipeline-loraqlora-fine-tuning)
- [Config-driven pipeline DPO/GRPO/PPO](#config-driven-pipeline-dpogrpoppo)
- [Config-driven pipeline multimodal (vision-language) SFT](#config-driven-pipeline-multimodal-vision-language-sft)
- [Examples](#examples)
- [Tests](#tests)
- [Profiling](#profiling)
- [API reference](#api-reference)
- [Known limitations](#known-limitations)

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

For `attn_implementation: flash_attention_2` on a Turing GPU (T4 etc.),
see `flash_attn_turing_shim/README.md` - the official `flash-attn` PyPI
package doesn't work in this kind of environment (see the training
infrastructure checklist below for why), so this is a real, separate
compatibility package, optional and only needed for that one config
field.

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

## Architecture

`ProcessGroupQUIC` (registered backend `"quic"`) sits directly under
`torch.distributed`; a vendored UDP hole-punch client establishes each
peer connection, handed off to a Rust-native, multi-channel QUIC driver
(one connection per peer, one named channel per `tag`) that owns the
real socket on its own background thread. See
[`docs/architecture.md`](docs/architecture.md) for the full component
diagram, the connection-establishment sequence, the Rust workspace
layout, and a callout on the two genuinely separate signaling-server
flows (hole-punch vs. the rendezvous key-value store) that share one
HTTP process but serve unrelated purposes.

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

Scaled up and CONFIRMED CROSS-MACHINE at real ~7B scale too:
`llava-hf/llava-1.5-7b-hf` (CLIP ViT-L/14 + Llama-7B, 32 layers, QLoRA,
`configs/llava_v15_7b_multimodal.yaml`) - a genuinely different decoder
family and vision tower than the 0.5B proof, module structure
reconfirmed by direct inspection rather than assumed. 36 real steps (3
epochs), loss 2.76 → 1.77 (avg first/last 8 steps), 79s total on the
cross-machine run. Two more real per-checkpoint gotchas found via live
runs at this scale (both now called out directly in the config's own
comments so they're not rediscovered): `image_token_id` differs per
checkpoint (151646 for the 0.5B Qwen proof, 32000 for this one - the
module's default matches only the first), and `compute_dtype` must
match the checkpoint's real native dtype (this one is float16, not the
0.5B proof's bfloat16) - both are real, direct crashes if wrong, not
silent degradations.

## Examples

19 rank scripts live in `examples/`, split into two families: most are
config-driven (`<config.yaml> <rank> <signaling_url> [job_id]`), a few
are hand-rolled pedagogical scripts with no config file. See
[`examples/README.md`](examples/README.md) for the full table of every
script (which family, exact usage, what it needs) and one complete,
copy-pasteable walkthrough end to end.

## Tests

```bash
cd tests
pip install pytest
python3 -m pytest -q
```

Most of these spin up a real local signaling server and real
hole-punched QUIC connections (loopback) — no mocking, and live in
`tests/` alongside their own `_helpers.py`. Two files are the
exception, pure CPU with no network/signaling server needed:
`test_checkpoint.py` (exercises `training_utils.py`'s checkpoint
save/load/resume directly) and `test_config_parsing.py` (round-trips
every training mode's config `.from_file()`). `examples/` holds the standalone,
directly-run scripts plus the `configs/` and `data/` they read from —
see [Examples](#examples) above — run one instance per machine/rank,
pointed at a real publicly reachable signaling URL.

## Profiling

`QUIC_DIST_RUST_DEBUG=1` is a real, working diagnostic that reports
live QUIC connection state (congestion window, RTT, loss, ACK/frame
counts) directly from quinn-proto — used throughout this project's own
bug-hunting history. See [`docs/profiling.md`](docs/profiling.md) for
how to enable it, what each field means, and a worked example of
distinguishing real network loss from a stalled driver loop.

## API reference

`quic_dist`'s public surface (`__init__.py`'s `__all__`):

| Name | What it is |
|---|---|
| `init_process_group(signaling_url, rank, world_size, job_id=, timeout=)` | Convenience wrapper: builds a `QuicRendezvousStore` and calls `dist.init_process_group(backend="quic", ...)`. The one function most users need. |
| `ProcessGroupQUIC` | The `torch.distributed.ProcessGroup` implementation itself, registered as backend `"quic"`. `send`/`recv`/`isend`/`irecv`/`barrier` only — see [Scope](#scope). |
| `QuicRendezvousStore` | A `torch.distributed.Store` backed by the signaling server's `/kv/*` API — used for rendezvous/barrier before any `ProcessGroup` exists. |
| `serialize_tensor(tensor, message_id, microbatch_id, tensor_id)` / `deserialize_tensor(payload)` | Tensor⇄bytes conversion `ProcessGroupQUIC` uses internally — exported for anyone building on the wire format directly. |
| `TensorMetadata` | The parsed-header type `deserialize_tensor` returns alongside the tensor. |
| `submit_as_work(fn)` | Runs `fn` on a background thread, returns a real `torch.distributed.Work` — what `isend`/`irecv` are built on. |

## Known limitations

- **Point-to-point only.** `all_reduce`/`all_gather`/`broadcast`/etc. all
  raise `NotImplementedError` by explicit design (see [Scope](#scope)) —
  not a silent no-op, not a fallback to another backend.
- **CPU-tensor-only serialization** — GPU tensors are moved to CPU
  before serialization and back after, at the `ProcessGroupQUIC`
  boundary.
- **One shared seed across every rank, by design** — reproducibility
  means "this exact run reproduces," not a decorrelated RNG stream per
  rank.
- **`rlhf.py`'s GRPO/PPO/RLOO/RM/PRM modes** don't have
  `pipeline_overlap_microbatches`/`overlap_communication` wired in yet
  (only `finetune.py`'s SFT/CPT path and `rlhf.py`'s DPO path do).
- **Parallel-stream send/recv is off by default**, even though fully
  fixed and validated — pending a product decision on the new default
  threshold, not a correctness concern. Opt in via
  `QUIC_DIST_PARALLEL_STREAM_THRESHOLD_BYTES`.
- **No formal mixed-precision (`torch.autocast`) wrapping.**
- **No standard external benchmark integration** beyond the in-repo
  held-out-slice eval.
- **`distill.py`/`multimodal.py`'s new checkpoint/seed/log wiring**
  (this session's polish pass) shares `pretrain.py`'s proven pattern and
  is syntax/import-verified, but wasn't individually GPU-smoke-tested
  the way `pretrain.py`'s was (a real kill-mid-run + resume test, see
  `docs/development-log.md`) - would need downloading a real
  teacher+student pair or a real LLaVA checkpoint, meaningfully more
  expensive than `pretrain.py`'s from-scratch tiny model.

Full real bug-fix and validation history — every item above, with the
actual root causes and measured before/after numbers — lives in
[`docs/development-log.md`](docs/development-log.md), out of this
file's way so this page stays readable as onboarding material.
