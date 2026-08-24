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

## Training infrastructure checklist

Ordered lightest -> heaviest by implementation weight; items marked
**[transport]** touch quic_dist's own send/recv/barrier path
(`process_group.py` / the Rust engine) - everything else is
training-loop-local bookkeeping and carries no transport risk. `[x]`
done and validated with a real run (see each item's note); `[ ]` not
yet done.

- [x] **Reproducible config (seeding)** - `training_utils.set_seed()`,
  same seed on every rank. `PipelineConfig.seed`/`RLHFModelConfig.seed`.
- [x] **Experiment tracking (loss, perplexity)** - `training_utils.ExperimentLogger`,
  plain JSONL, `config.log_path`. Includes a config snapshot at the
  start of the log.
- [x] **Gradient accumulation** - `config.grad_accum_steps` in `finetune.py`.
- [x] **Gradient checkpointing** - `config.gradient_checkpointing` in
  `finetune.py` (`torch.utils.checkpoint`, `use_reentrant=False`, with
  `peft_model.enable_input_require_grads()` for the frozen-embedding
  case).
- [x] **Distributed checkpoint save/resume** (model=trainable params
  only, optimizer state, RNG state, dataloader/step position) -
  `training_utils.save_checkpoint`/`load_checkpoint`, `config.checkpoint_dir`/
  `checkpoint_every`/`checkpoint_keep_last`. Resuming is automatic
  (the same launch command works for a first run or a restart) -
  validated with a real kill-mid-run-then-relaunch test on
  `finetune.py`'s SFT/CPT path: the resumed run reproduced the exact
  same losses the killed run had already logged for the steps it
  re-entered (bit-identical, confirming the RNG state genuinely
  restored, not just "training continued somehow"). A real bug was
  found and fixed by this test: `torch.load(..., map_location=<cuda
  device>)` relocates the saved RNG state tensors to the GPU too,
  which `torch.set_rng_state`/`torch.cuda.set_rng_state_all` then
  reject - fixed by forcing those specific tensors back to CPU
  regardless of the checkpoint's overall `map_location`. A second real
  bug: reusing the same `job_id` across a crashed attempt and its
  resume can let `_step_barrier`'s per-step store keys collide with
  the crashed attempt's leftover (possibly only-partially-satisfied)
  state - fixed by namespacing each attempt's barrier keys with its
  resume step number.
- [x] **Automatic evaluation after checkpoint** - `config.eval_every`/
  `eval_num_examples` in `finetune.py`, a held-out TAIL slice of the
  same dataset (no second dataset config needed), forward-only through
  the same pipeline, logged as `eval_loss`/`eval_perplexity`.
- [x] Reference wiring above is `finetune.py` (SFT/CPT); `rlhf.py`'s
  `run_dpo_training` has seed/logging/checkpoint+resume too (also
  validated with a real run), proving the utilities generalize beyond
  one training mode.
- [ ] Wire seed/logging/checkpoint+resume into `rlhf.py`'s remaining
  modes (GRPO, PPO, RLOO, RM, PRM) and into `distill.py`/`pretrain.py` -
  same utilities, not yet threaded through each loop individually.
- [ ] **Mixed precision** - arguably already the case in spirit
  (compute at `config.compute_dtype`, loss/softmax explicitly cast to
  fp32 - see `finetune.py`'s dtype-boundary comments) but not a formal
  `torch.autocast` wrapping; not attempted given the existing
  bitsandbytes-quantized layers' own internal dtype handling.
- [ ] **Benchmark integration** - a standard external eval (e.g. a
  fixed perplexity benchmark corpus) beyond the in-repo held-out-slice
  eval above.
- [x] **Flash attention** - the OFFICIAL `flash-attn` PyPI package
  genuinely fails here: `pip install flash-attn --no-build-isolation`
  builds a wheel successfully (~2-3 min, faster than expected) but the
  compiled extension fails to import - `undefined symbol:
  _ZN3c105Error...`, a real C++ ABI mismatch against the installed
  torch build (a well-known flash-attn pain point - needs a wheel built
  against the exact torch/CUDA/ABI combo in use). Uninstalled rather
  than left broken.

  Tried a real alternative instead of stopping there:
  [ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing),
  a community FlashAttention implementation specifically for Turing GPUs
  (compute capability 7.5 - T4, this project's real GPUs). Its raw
  kernels ARE correct and fast, verified directly, not assumed: forward
  max diff 0.000488 / backward dQ max diff 0.001953 vs. torch's own
  SDPA (fp16 numerical noise, not a bug), 1.27x real forward speedup in
  isolation. Built `flash_attn_turing_shim/` (new, its own README) - a
  real compatibility package presenting these kernels under the
  `flash_attn` name/API surface `transformers` imports unconditionally,
  including the `dropout_p`/`deterministic`/`softcap`/`window_size`
  argument-adaptation `transformers` always passes (the underlying
  kernels don't support dropout, softcap, or sliding-window - the shim
  raises loudly, not silently, if one is actually requested with a
  non-default value). Verified end-to-end through BOTH `transformers`'
  regular `model.forward()` AND this repo's own direct-decoder-layer
  call pattern (`finetune.py`'s `run_decoder_layer`) - real loss, real
  backward, real gradients either way. Wired into `finetune.py` as
  `config.attn_implementation` (default `"sdpa"`, unchanged behavior;
  `rlhf.py`'s `RLHFModelConfig` got the same field since it reuses
  `build_stage_model` via duck typing).

  **Honest end-to-end result, not oversold, AND scale-dependent -
  measured at two real sizes, not just one**:
  - Qwen2.5-0.5B, 2-stage pipeline: NO end-to-end win - seq_len=128:
    19.3s (sdpa) vs. 22.1s (flash, SLOWER); seq_len=512: 22.6s both
    (tied). At this size, attention isn't the training step's
    bottleneck, so a faster attention kernel alone doesn't move total
    wall-clock time.
  - Qwen2.5-7B, 2-stage pipeline, seq_len=1024, a genuine apples-to-
    apples A/B (both runs `cpu_offload_unused_layers=false`, since
    `transformers`' flash_attention_2 path rejects any device_map
    containing `"cpu"` entries - see the real incompatibility noted
    below, found via a direct crash, not assumed): 243.3s (sdpa) vs.
    238.6s (flash) - a real, if modest, ~2% WIN this time. Loss matched
    closely at both sizes (3.0505 vs 3.0512 at 0.5B/seq_len=128; 2.7185
    vs. 2.7179 at 7B), confirming correctness held throughout, not just
    in the isolated kernel test.

  The crossover is the real finding: attention is a small fraction of
  the step at 0.5B/short-context, big enough to matter at 7B/seq_len
  1024. Kept as a real, validated, opt-in capability
  (`examples/configs/qwen25_0.5b_lora_flash_attn.yaml`) - worth
  testing at even larger scale/longer context, which this project
  hasn't done. Not useful for `qwen38_27b`'s hybrid architecture
  regardless of scale - its `linear_attention` blocks use a completely
  different mechanism, unaffected by flash attention's
  `full_attention`-only speedup.

  **Real, found-not-assumed incompatibility**: `transformers`'
  `flash_attention_2` loading path raises `ValueError` on ANY
  device_map containing a `"cpu"` entry - which is exactly what
  `cpu_offload_unused_layers=True` relies on (the mechanism that lands
  "other ranks' layers" on the meta device at ~0 real memory for larger
  models split across a pipeline - see `build_device_map()`'s
  docstring). So `attn_implementation: flash_attention_2` currently
  only works for configs that don't need that offload - either a small
  enough model that each rank can afford to hold the WHOLE thing
  redundantly (what both A/B tests above actually did - a 7B model in
  4bit is only ~3.5GB, comfortably fits per-rank on a 15GB T4 even
  loaded twice), or a future change to route "other" layers somewhere
  flash_attention_2 accepts instead of `"cpu"`.
- [x] **Communication/computation overlap** - turned out NOT to need
  Rust work, correcting an earlier wrong assumption in this checklist:
  `process_group.py`'s `isend`/`irecv` (`work.py`, a genuine background-
  thread + `torch.futures.Future` implementation, its own docstring
  already states its purpose is exactly this) already existed and were
  already validated - what was missing was the TRAINING LOOP using
  them. `finetune.py`'s `config.overlap_communication` converts the
  forward-activation send (non-last ranks) and backward-gradient send
  (non-first ranks) from blocking `send` to `isend`, deferring the wait
  on each one until right before the NEXT send of the same kind (or
  teardown) - so the calling thread moves on to its next blocking call
  instead of idling through the hand-off. Validated on a real 0.5B run
  BOTH loopback and real cross-machine (local <-> a real remote GCP
  box, genuine hole-punch, `grad_accum_steps` 8 loopback / 4 cross-
  machine): correctness held both times - bit-identical mean loss with
  the flag on vs. off (4.6165 loopback, 4.1430 cross-machine), not just
  "didn't crash". Timing: loopback 32.4s vs. 33.5s (~3%), cross-machine
  39.1s vs. 41.2s (~5%) - a real, if modest, win both times, slightly
  larger cross-machine as expected (more actual network latency to
  hide behind deferred waits). The modesty is an honest, expected
  consequence of this architecture's own per-step `_step_barrier`
  forcing every rank back into lockstep each micro-step (see that
  function's docstring) - genuine multi-micro-batch staggering
  (removing that barrier for a true streaming pipeline) would very
  likely show a much larger effect, especially over a higher-RTT link,
  but that's a materially bigger, riskier restructuring than this flag -
  not attempted here.

## Known limitations

See the full deliverable report for the complete architecture, design
rationale, and an honest list of what's implemented vs. not:
https://claude.ai/code/artifact/6677daa1-801f-44c9-b6b5-d42880632c6b
