# Development log: training infrastructure checklist

Real bug-fix and validation history for the training-infrastructure work
on top of `quic_dist`'s transport layer — moved out of the top-level
`README.md` so that file can read as onboarding material; nothing here
is summarized or trimmed, it's a verbatim move. See `README.md` for the
newcomer-facing overview and `docs/architecture.md` for how the pieces
fit together.

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
- [x] Wired seed/logging/checkpoint+resume into `distill.py`/
  `pretrain.py`/`multimodal.py` too - see the "Repo polish pass" entry
  near the end of this file for the full details, including a real bug
  this wiring surfaced and fixed (pretrain.py's from-scratch model
  never restricted `requires_grad`, which would have made checkpoint
  save silently include the WHOLE model instead of just each rank's own
  slice). `rlhf.py`'s remaining modes (GRPO, PPO, RLOO, RM, PRM) still
  don't have it individually threaded through, since they don't share
  `run_dpo_training`'s loop.
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
  attempted below, not left as a TODO.

- [x] **Real multi-micro-batch pipeline overlap** (`config.pipeline_overlap_microbatches`)
  - the actual fix for `overlap_communication`'s own documented
  ceiling above: this one removes the per-micro-batch `_step_barrier`
  and instead runs a real GPipe-style schedule per accumulation window
  (`finetune.run_gpipe_window` - all forwards for the window first,
  async `isend` + one-ahead `irecv` prefetch, THEN all backwards, THEN
  one `optimizer.step()`) - requires `grad_accum_steps > 1` to do
  anything, and only ONE `_step_barrier` per WINDOW instead of per
  micro-batch. Still no Rust work - same transport primitives as
  `overlap_communication` above, just used across MULTIPLE
  concurrently-in-flight micro-batches instead of one at a time.

  **A real bug found and fixed during validation, not just written and
  assumed correct**: the first end-to-end test hung, deterministically,
  on the LAST window of an 8-window run - rank0 stuck waiting on a
  gradient that never arrived, while rank1 had already finished
  cleanly. Root cause, found by reading the Rust engine's source, not
  guessed: `multiplexed_driver.rs` keys each message tag string to a
  **persistent, reusable channel/stream**
  (`out_channels: HashMap<String, OutboundChannel>`, one pending-send
  slot, not a fresh stream per message) - safe for the per-micro-batch
  loop's strictly-sequential one-tag-at-a-time usage, but NOT safe for
  this scheduler's overlapping usage: window 8 reused window 7's exact
  tag set (`step_counter % 8` repeats every 8 steps), and a new
  window's first send/recv on a tag could start before the previous
  window's use of that same tag was fully settled on both sides, not
  just locally `.wait()`-ed. The original tag scheme also reused the
  SAME tag for the forward-hidden channel and the backward-gradient
  channel - a second real instance of the same mistake. Fixed by making
  every tag globally unique per (direction, micro-batch) for the whole
  run, sidestepping the whole class of bug rather than fully
  characterizing the exact Rust-side race. Re-tested 3 times after the
  fix (including the exact window-7-to-8 boundary that hung before) -
  clean every time.

  **Validated results, loopback AND real cross-machine** (Qwen2.5-0.5B,
  2-stage pipeline, `grad_accum_steps=8` loopback / `4` cross-machine):
  correctness - loss bit-identical to the non-overlap path at every
  matching step, both settings. Timing: loopback 33.1s -> ~23-25s across
  3 repeated runs (**~28% faster**); cross-machine (local <-> a real
  remote GCP box) 39.9s -> 18.7s (**~53% faster, more than 2x**) - a
  dramatically bigger win than `overlap_communication` alone, and
  bigger cross-machine than loopback as expected (real network latency
  is exactly what this schedule hides behind compute, unlike loopback's
  near-zero latency).

  **Real cost, not free**: peak activation memory scales with
  `grad_accum_steps` - every micro-batch's activation in the window
  stays alive in GPU memory until ITS backward runs, not just one at a
  time. `examples/configs/qwen25_0.5b_lora_pipeline_overlap.yaml` is a
  real, validated example. Only wired into `finetune.py`'s SFT/CPT path
  so far, not into `rlhf.py`'s DPO/GRPO/PPO/RLOO/RM/PRM.

- [x] **Parallel-stream send/recv stall - resolved** (`ProcessGroupQUIC.
  _use_parallel_streams`/`_chunk_plan` in `process_group.py`) - large
  tensors fan out over several concurrent QUIC streams instead of one.
  Previously shipped OFF by default (threshold effectively unreachable)
  after a real, then-unresolved stall on repeated large sends over one
  connection. This session found and fixed THREE independent, real root
  causes (none of them the "Cubic app_limited" theory the code's own
  prior comment spent most of its words on - that theory was plausible
  from the stats alone but wasn't the actual mechanism):

  1. Linux doubles whatever `SO_RCVBUF`/`SO_SNDBUF` value it grants
     before returning it from `getsockopt()` (kernel bookkeeping, not
     real payload capacity) - `window` in `process_group.py` used the
     raw (doubled) value, confirmed directly (`getsockopt` returned
     exactly 2x `net.core.rmem_max` on this project's own machines).
  2. `max_congestion_window` was ALSO set to `8 * window` (the same
     ratio as the flow-control windows) instead of `1 * window` - the
     very ratio this project's own upstream
     (`vllm/transport/quic_transport.py`) documents as correct ("capped
     at 1x - not 8x") but whose own code, three lines below that
     comment, still passes 8x - a real, pre-existing comment/code drift
     inherited here unnoticed, caught by this session's direct repro.
     Combined with #1, the effective cap was ~16x the real kernel
     buffer, so `congestion.rs`'s `BoundedController` (built to prevent
     exactly this) never actually engaged before self-induced loss did.
  3. A genuine driver-loop bug in `multiplexed_driver.rs::
     drive_channel_send`: a PARTIAL write (`write_stream` returning
     `Ok(n)` with `n` < requested - a real limit hit, not a full block)
     never set `blocked_on_writable`, so nothing ever re-drove that
     channel unless a fresh `Writable` edge happened to fire on its own.
     `driver.rs`'s older single-channel loop avoids this by retrying
     unconditionally every tick; the multi-channel port never got the
     same treatment - a channel could stall forever with data still
     queued, zero loss, full peer ACK coverage, and free congestion-
     window room, simply because nothing ever asked it to keep writing.

  Fixing this also surfaced a real regression in `_get_or_connect`'s
  auto-reconnect race (`ProcessGroupQUIC._retry_dead_conn`, new this
  session): `is_dead()` is only checked BEFORE handing a connection to
  the caller, so a connection dying WHILE a blocked `recv()` call is
  in flight still raised the stale connection's own error instead of
  transparently reconnecting - always possible, just rare enough that
  the original (slower) driver loop's timing had apparently never hit
  it in this test suite before. Fixed with a bounded one-retry wrapper.

  **Validated results**: loopback (3 repeated runs of a back-to-back
  large-message repro, plus an 8-message mixed-size stress run up to
  32MB/message, byte-exact every time); the full pytest suite (19/19)
  on two independent machines sharing the identical 208KB
  `net.core.rmem_max`/`wmem_max` ceiling; and a REAL cross-machine run
  over an actual network path (~1ms measured RTT, not loopback) for
  both the basic and stress repros, byte-exact, no stall. Still off by
  default pending a decision on the new default threshold - opt in via
  `QUIC_DIST_PARALLEL_STREAM_THRESHOLD_BYTES`.

- [x] **Repo polish pass** (tests, documentation, architecture diagram,
  clean API, reproducible examples, benchmark, profiling, checkpointing)
  - the final tidy-up before considering the repo done, not new
  capability. Bounded, surgical fixes over risky refactors throughout -
  see each bullet for what was actually verified, not just written:

  - **Tests**: `tests/test_checkpoint.py` (new, 4 tests) closed
    checkpointing's previously-zero coverage - save/load/resume
    round-trip (trainable params, optimizer momentum/variance, and RNG
    state all restored bit-for-bit into a *fresh* model/optimizer
    instance, not the one that saved them), `keep_last` pruning,
    per-rank isolation, and the documented "nothing to resume" `None`
    contract. `tests/test_config_parsing.py` (new, 28 tests) round-trips
    every training mode's config `.from_file()` (YAML and JSON), and
    doubles as a regression guard on the new fields below (asserts their
    defaults on every config class, and that they're actually
    overridable from a file, not silently ignored). Full suite: 19 -> 51
    passing.
  - **Documentation**: this file is the result - the old README's
    "Training infrastructure checklist" section (57% of the file) moved
    here verbatim, so the top-level README reads as onboarding material
    instead of a changelog. Added a table of contents.
  - **Architecture diagram**: `docs/architecture.md` - real, in-repo
    Mermaid (renders natively on GitHub): a component diagram, a
    connection-establishment sequence diagram, an explicit callout on
    the two genuinely separate signaling-server flows (hole-punch vs.
    the `QuicRendezvousStore` rendezvous key-value store) that share one
    HTTP process but serve unrelated purposes, and the Rust workspace
    layout - including documenting `driver.rs`/`PyQuicConnectionDriver`/
    `PyQuicEngine`'s real status (intentionally kept as a reference
    implementation used to diff real bugs against, e.g. the parallel-
    stream stall entry above - not dead code). Replaces a single link to
    a private, inaccessible external artifact the README used to end on.
  - **Clean API**: `__init__.py`'s public surface was already clean (no
    dead exports) - the real gap was `distill.py`/`pretrain.py`/
    `multimodal.py` having ZERO of `seed`/`attn_implementation`/
    `checkpoint_dir`/`checkpoint_every`/`checkpoint_keep_last`/
    `log_path`, unlike `finetune.py`/`rlhf.py`. Added the exact same
    fields (copied, not reinvented) to all three, wired into each run
    function the same proven way. `distill.py` needed a genuine
    variation, not a blind copy: teacher/student get INDEPENDENT
    `teacher_attn_implementation`/`student_attn_implementation` fields
    (matching its existing `teacher_quantization`/`student_quantization`
    split), and only the student (rank 1) checkpoints at all (the
    teacher is frozen/inference-only, no optimizer state to save) - which
    surfaced a real correctness requirement: the teacher has no
    checkpoint of its own to read a resume step from, so without an
    explicit resume-step exchange between ranks, a resumed run would
    have the teacher replay every step from 0 while the student skipped
    ahead, desyncing their tag sequence and hanging the per-step
    barrier permanently. Fixed with a small dedicated `dist.send`/`recv`
    exchange (its own tag) right after the checkpoint load. A second
    real bug, in `pretrain.py`: nothing in `build_pretrain_stage_model`
    ever restricted `requires_grad` - this is a from-scratch model with
    no LoRA/frozen backbone, so every parameter (including the ones
    sitting unused on OTHER ranks' CPU, per this module's own "build the
    whole model, keep only this rank's slice" design) defaulted to
    `requires_grad=True`. Without a fix, checkpoint save would have
    silently included the WHOLE model's random-init weights per rank,
    not just this rank's own pieces. Fixed by freezing everything first,
    then re-enabling gradients only on the pieces actually moved to this
    rank's GPU. `pretrain.py`'s `attn_implementation` on
    `AutoModelForCausalLM.from_config()` (not `from_pretrained()`, since
    there's no checkpoint) was verified empirically, not assumed, to
    accept the kwarg identically on this project's installed
    `transformers` version.
  - **Reproducible examples**: `cli.py` (new, root-level, imports as
    `quic_dist.cli`) extracts the ~10-line argv-parsing+bootstrap block
    8 of the 10 config-driven `examples/*_pipeline_rank.py` scripts had
    each hand-copied identically, down to one function call per script
    (`grpo_pipeline_rank.py`/`ppo_pipeline_rank.py` have real extra
    logic - optional reward-model loading - so they're left parsing
    `sys.argv` directly, not forced through the helper). Verified with a
    real functional check (not just import) of `run_pipeline_rank_main`
    against a fake config/run_fn, confirming both the default- and
    explicit-`job_id` argv-parsing paths. `examples/README.md` (new)
    tables all scripts (which family, exact usage, what's needed),
    calling out `lora_pipeline_rank.py`'s misleading name (looks
    config-driven like its neighbors, is actually a from-scratch
    hand-rolled pedagogical example with no config file at all).
  - **Benchmark**: `examples/bench_quic_dist.py` had a real, confirmed-
    broken import (`sys.path` never added `tests/`, where `_helpers.py`
    actually lives - it crashed immediately with `ModuleNotFoundError`
    on a fresh checkout, verified by actually running it before AND
    after the fix, not just reading the diff).
  - **Profiling**: `docs/profiling.md` (new) documents
    `QUIC_DIST_RUST_DEBUG=1` (real, already used throughout this file's
    own bug-hunting history above) as actual user-facing tooling for the
    first time - what each `debug_stats()` field means, and the specific
    sender-vs-receiver-byte-count comparison that distinguishes real
    network loss from a stalled driver loop (the exact technique that
    found the parallel-stream stall's three root causes above). No new
    tooling - the diagnostic already worked, it just wasn't documented.
  - **Checkpointing**: beyond the config/wiring work above, real
    end-to-end validation, not just "looks right" - a genuine kill-mid-
    run-then-resume test on `pretrain.py` (`examples/configs/
    pretrain_tiny_qwen2_ckpt_test.yaml`, new): ran a full 32-step
    reference run, then a second run killed (`kill -9`, both ranks) right
    after the step-6 checkpoint, then resumed with the SAME launch
    command. Every resumed step (7 through 16) reproduced the reference
    run's loss BIT-FOR-BIT (e.g. step 11: 11.1182 both times, to full
    float precision), confirming the seed/RNG-state/optimizer-state/
    barrier-namespacing machinery all actually works together, not just
    each piece in isolation. `distill.py`/`multimodal.py` got the same
    proven wiring pattern and are syntax/import-verified, but weren't
    individually GPU-smoke-tested the same way (would need downloading a
    real teacher+student pair or a real LLaVA checkpoint + image
    dataset - meaningfully more expensive than pretrain.py's from-
    scratch tiny model, which needs no checkpoint download at all) -
    worth a real run before relying on either in production, same
    honesty standard as every other item in this file.

- [x] **External-rollout GRPO entry point** (`rlhf.run_grpo_training_from_rollouts`,
  `rlhf.RolloutBatch`) - the integration surface for `quic-rl`, a
  separate orchestration repo built on top of this project and
  `quic-vllm` for online RL (quic-vllm generates rollouts -> quic-rl
  scores them -> this project trains -> quic-rl syncs the updated policy
  back). `run_grpo_training` used to be the ONLY GRPO entry point and was
  monolithic: it called `pipeline_generate` itself, in-process, using
  this project's own pipeline-parallel model - there was no way to feed
  it externally-generated `(prompt, response, reward)` triples instead.

  Rather than duplicate GRPO's actual loss math into the new repo (the
  explicit thing that integration was designed NOT to do), the existing
  loop's loss-computation body - teacher-forced ref+policy forward,
  group-relative advantage, `pg_loss`/KL, backward, `optimizer.step()` -
  was factored out into `_grpo_update_from_rollout`, a shared helper both
  `run_grpo_training` and the new `run_grpo_training_from_rollouts` call.
  `run_grpo_training`'s own behavior, call sites, and tests are
  completely unchanged - same function, same tag scheme, same barrier
  discipline, verified with a real 2-rank Qwen2.5-0.5B run (16/16 GRPO
  steps, real loss/reward/KL trajectory) after the refactor, not just
  read-through.

  `run_grpo_training_from_rollouts` takes any iterable of `RolloutBatch`
  (`prompt_ids`/`generated`/`rewards` - the exact shapes
  `pipeline_generate` already produces, so nothing downstream needs to
  know which path produced them) instead of calling `pipeline_generate`
  - every rank's process gets an equivalent `rollout_source`, matching
  this project's own existing "every rank redundantly builds the
  identical data locally" pattern rather than one rank broadcasting to
  the others. Validated with a real, separate 2-rank run (same model,
  synthetic-but-real rollout data with a deliberately-non-uniform reward
  per group member): real backward/optimizer.step() every step, real
  non-NaN loss values, and `reward_mean` in the logs matched the exact
  mean of the synthetic rewards handed in - direct confirmation the
  reward data flows through the group-relative advantage computation
  correctly, not just "didn't crash."

  No seed/checkpoint/log wiring added to either GRPO path as part of
  this - GRPO didn't have that wired in before this change either (see
  the earlier "wire seed/logging/checkpoint+resume into rlhf.py's
  remaining modes" item, still open); adding it is real, separate future
  work, not something this integration point should invent as a side
  effect.

## Known limitations

Consolidated here (used to be a single broken external link):

- **Point-to-point only.** `all_reduce`/`all_gather`/`broadcast`/etc. all
  raise `NotImplementedError` by explicit design (see `README.md`'s
  "Scope" section) - not a silent no-op, not a fallback to another
  backend. Only `send`/`recv`/`isend`/`irecv`/`barrier` work.
- **CPU-tensor-only serialization.** `tensor.py`'s `serialize_tensor`/
  `deserialize_tensor` operate on CPU tensors; GPU tensors are moved to
  CPU before serialization and back after deserialization at the
  `ProcessGroupQUIC` boundary - real, not a hidden perf cliff, but worth
  knowing before profiling an unexpected host-device copy.
- **One shared seed across every rank, by design** (`training_utils.
  set_seed`'s own docstring) - reproducibility here means "this exact
  run reproduces," not "each rank gets a decorrelated RNG stream."
- **`rlhf.py`'s GRPO/PPO/RLOO/RM/PRM modes** don't have
  `pipeline_overlap_microbatches`/`overlap_communication` wired in yet
  (only `finetune.py`'s SFT/CPT path and `rlhf.py`'s DPO path do).
- **Parallel-stream send/recv is off by default** even though fully
  fixed and validated this session (see the entry above) - pending a
  product decision on the new default threshold, not a correctness
  concern. Opt in via `QUIC_DIST_PARALLEL_STREAM_THRESHOLD_BYTES`.
- **No formal mixed-precision (`torch.autocast`) wrapping** - see the
  "Mixed precision" item above for why this wasn't attempted as-is.
- **No standard external benchmark integration** (e.g. a fixed
  perplexity benchmark corpus) beyond the in-repo held-out-slice eval.
