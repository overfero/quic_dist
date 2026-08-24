"""Config-driven pipeline-parallel LLM PRE-training FROM SCRATCH, sibling
to finetune.py (SFT/CPT on an EXISTING checkpoint), rlhf.py, multimodal.py,
and distill.py.

Architecturally the one real difference from every other module here:
every other mode calls `AutoModelForCausalLM.from_pretrained(...)` -
real pretrained weights, then adapted (LoRA/QLoRA) or distilled from.
This module calls `AutoModelForCausalLM.from_config(...)` instead - a
model built from an architecture SPEC with randomly-initialized
weights, nothing loaded from a checkpoint at all. That single API
difference cascades into everything else being different too:

- No LoRA, no quantization. There is no frozen pretrained backbone to
  preserve behind a small adapter - pretraining trains EVERY parameter
  of the (small, real) model. bitsandbytes quantization is specifically
  a way to keep a large frozen checkpoint cheap in memory; quantizing
  freshly-initialized random weights you're about to train from scratch
  makes no sense and would only hurt optimization.
- `build_pretrain_stage_model()` builds the FULL model (small enough to
  construct entirely, even if wastefully replicated on every rank's CPU
  RAM - see PretrainConfig's docstring for real, concrete size
  guidance) then moves ONLY this rank's own pieces (its slice of
  `layers`, plus `embed_tokens`/`rotary_emb`/`norm`/`lm_head` where
  relevant) onto its GPU - everything else stays on CPU, unused. This
  is deliberately simpler than finetune.py's quantization-based
  `cpu_offload_unused_layers` (there's no bitsandbytes meta-device trick
  available here since nothing is quantized) - correct and safe ONLY
  because the model is kept genuinely small (see the RAM-exhaustion
  lesson from this project's own history: replicating even a modest
  model on every rank's CPU is fine, replicating anything checkpoint-
  sized is not - this module has no answer for a large from-scratch
  pretrain, and doesn't pretend to).
- The tokenizer is NOT trained from scratch - an existing real
  tokenizer (`tokenizer_path`, any public checkpoint's) is reused. A
  from-scratch BPE/tokenizer training pipeline is a genuinely separate,
  substantial piece of work, out of scope here; reusing an existing
  vocabulary is standard practice for a from-scratch PRETRAINING proof
  at small scale.
- The architecture itself is built via `transformers.AutoConfig.for_model
  (architecture, **hyperparameters)` - confirmed directly (not assumed)
  that this real, documented HF factory produces a working config for
  an unloaded checkpoint, then `AutoModelForCausalLM.from_config(cfg)`
  builds a real model against it with random weights. Defaults to
  "qwen2" so `layers_attr` etc.'s plain defaults (`model.layers` etc.,
  same as finetune.py's) apply unmodified.

Dataset is always raw-text continuation (no instruction/response
masking - that concept doesn't apply to pretraining) - see
`build_pretrain_dataset`'s own docstring for the real
`load_dataset(...)`-shape issue (wikitext requires a config name; many
rows are blank/section-header lines) this handles.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import quic_dist
import torch.distributed as dist

from quic_dist.finetune import resolve_attr, stage_range, run_decoder_layer, _step_barrier


@dataclass
class PretrainConfig:
    tokenizer_path: str
    world_size: int
    num_layers: int
    stage_layer_counts: list[int] | None = None

    layers_attr: str = "model.layers"
    embed_attr: str = "model.embed_tokens"
    norm_attr: str = "model.norm"
    rotary_attr: str | None = "model.rotary_emb"
    lm_head_attr: str = "lm_head"

    # Architecture spec for transformers.AutoConfig.for_model() - a REAL
    # config class gets built and used, just with random weights instead
    # of a checkpoint. Defaults deliberately tiny: kept small enough
    # (~tens of millions of non-embedding params) that replicating the
    # FULL model on every rank's CPU RAM (see build_pretrain_stage_model)
    # stays cheap - this is not a config to casually scale up without
    # revisiting that design, see this module's own docstring.
    architecture: str = "qwen2"
    vocab_size: int | None = None  # None -> tokenizer's own vocab_size
    hidden_size: int = 512
    num_attention_heads: int = 8
    num_key_value_heads: int | None = None  # None -> = num_attention_heads (no GQA reduction)
    intermediate_size: int = 1376
    max_position_embeddings: int = 512
    compute_dtype: str = "bfloat16"

    dataset_name: str = "Salesforce/wikitext"  # namespaced repo id - see examples/configs/pretrain_tiny_qwen2.yaml's
                                                # comment for the real HfUriError the bare "wikitext" name hits on
                                                # some datasets/huggingface_hub version combos
    dataset_config: str | None = "wikitext-103-raw-v1"  # load_dataset's `name=` - wikitext needs one
    dataset_split: str = "train"
    num_examples: int | None = 64
    text_field: str = "text"
    min_text_chars: int = 20  # wikitext has many blank/section-header rows - see build_pretrain_dataset

    seq_len: int = 96
    batch: int = 1
    epochs: int = 3
    lr: float = 3e-4  # pretraining from a random init typically wants a higher LR than fine-tuning an
                       # existing checkpoint (nothing to preserve, no risk of catastrophic forgetting)
    log_every: int = 8
    connect_timeout_s: int = 300

    # Reproducibility - same seed on every rank deliberately, see
    # training_utils.set_seed's docstring.
    seed: int = 42

    # Attention implementation, passed straight through to
    # AutoModelForCausalLM.from_config - see finetune.py's
    # PipelineConfig.attn_implementation for the full story
    # (flash_attention_2 needs flash_attn_turing_shim/ on Turing GPUs).
    # Confirmed empirically (not assumed) that from_config accepts this
    # kwarg identically to from_pretrained on this project's installed
    # transformers version.
    attn_implementation: str = "sdpa"

    # Checkpoint save/resume - every rank has real trainable params here
    # (unlike distill.py's frozen-teacher split), so this applies
    # uniformly, same contract as finetune.py's identical fields.
    checkpoint_dir: str | None = None
    checkpoint_every: int = 0  # steps; 0 = never checkpoint even if checkpoint_dir is set
    checkpoint_keep_last: int = 2

    # Experiment tracking - plain JSONL, see training_utils.ExperimentLogger.
    log_path: str | None = None

    @classmethod
    def from_file(cls, path: str) -> "PretrainConfig":
        text = Path(path).read_text()
        if path.endswith((".yaml", ".yml")):
            import yaml

            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        return cls(**data)

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.compute_dtype)


def build_pretrain_stage_model(rank: int, local_gpu: int, config: PretrainConfig):
    """Returns (model, stage_layers, embed_tokens, norm, rotary_emb_or_None,
    lm_head, hidden_size, trainable_params). Unlike finetune.py's
    build_stage_model, there is no peft wrapper and no device_map -
    the whole (small) model is built on CPU via from_config, then only
    this rank's own pieces are moved to its GPU; `trainable_params` is
    exactly those pieces' parameters (not `model.parameters()` - the
    rest of the model, sitting unused on CPU, must never enter this
    rank's optimizer)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as hflog

    hflog.disable_progress_bar()
    hflog.set_verbosity_error()

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.for_model(
        config.architecture,
        # len(tokenizer), NOT tokenizer.vocab_size - a real CUDA
        # "indexSelectLargeIndex ... srcIndex < srcSelectDimSize"
        # assertion is what this fixes: tokenizer.vocab_size (151643 for
        # Qwen2.5's tokenizer, confirmed directly) covers only the BASE
        # vocab, while len(tokenizer) (151665) includes special/added
        # tokens (eos/pad among them) that real tokenized text - and the
        # pad-token fallback this module already uses - can and does
        # emit. Building the embedding table to the smaller number
        # crashes the moment any such token appears.
        vocab_size=config.vocab_size or len(tokenizer),
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads or config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
    )
    model = AutoModelForCausalLM.from_config(model_config, torch_dtype=config.torch_dtype, attn_implementation=config.attn_implementation)

    device = torch.device(f"cuda:{local_gpu}")
    is_first = rank == 0
    is_last = rank == config.world_size - 1
    my_range = stage_range(rank, config)

    layers = resolve_attr(model, config.layers_attr)
    embed = resolve_attr(model, config.embed_attr)
    norm = resolve_attr(model, config.norm_attr)
    rotary = resolve_attr(model, config.rotary_attr) if config.rotary_attr else None
    lm_head = resolve_attr(model, config.lm_head_attr)

    # Real gap found while wiring checkpoint save in: nothing in this
    # function ever restricted requires_grad, and this model has no
    # naturally-frozen backbone (no LoRA, no quantization, no
    # from_pretrained) - EVERY parameter defaults to requires_grad=True,
    # including the pieces of the model that stay on CPU, unused, and
    # never touched by this rank at all (see this function's own
    # docstring). `training_utils.save_checkpoint` saves exactly the
    # params with requires_grad=True, so without this fix a checkpoint
    # would silently include the WHOLE model's random-init weights per
    # rank, not just this rank's own slice - defeating the "trainable-
    # params-only, stays small" design `training_utils.py`'s module
    # docstring describes. Freeze everything first, then re-enable
    # gradients only on the pieces actually moved to this rank's GPU.
    model.requires_grad_(False)

    stage_layers = layers[my_range.start : my_range.stop]
    trainable = []
    for layer in stage_layers:
        layer.to(device)
        layer.requires_grad_(True)
        trainable += list(layer.parameters())
    if is_first:
        embed.to(device)
        embed.requires_grad_(True)
        trainable += list(embed.parameters())
    if is_last:
        norm.to(device)
        lm_head.to(device)
        norm.requires_grad_(True)
        lm_head.requires_grad_(True)
        trainable += list(norm.parameters()) + list(lm_head.parameters())
    if rotary is not None:
        rotary.to(device)
        rotary.requires_grad_(True)
        trainable += list(rotary.parameters())  # usually none - rotary embeddings are buffers, not
                                                  # learnable weights - but harmless/correct either way

    print(f"[rank {rank}] trainable params: {sum(p.numel() for p in trainable)}, layers {list(my_range)}", flush=True)

    hidden_size = model_config.hidden_size
    return model, stage_layers, embed, norm, rotary, lm_head, hidden_size, trainable, tokenizer


def build_pretrain_dataset(tokenizer, config: PretrainConfig) -> torch.Tensor:
    """Raw-text next-token continuation - the only mode that makes
    sense for pretraining (no prompt/response masking concept).
    wikitext (the default corpus) needs its `name=` config argument, and
    a real chunk of its rows are blank lines or bare section headers
    (e.g. " = History = \\n") - both handled here rather than left as a
    silent trap: `dataset_config` is passed through if set, and rows
    shorter than `min_text_chars` are filtered out (over-fetching 3x
    `num_examples` raw rows first so filtering still leaves enough)."""
    from datasets import load_dataset

    fetch_n = None if config.num_examples is None else config.num_examples * 3
    split = config.dataset_split if fetch_n is None else f"{config.dataset_split}[:{fetch_n}]"
    if config.dataset_config:
        ds = load_dataset(config.dataset_name, config.dataset_config, split=split)
    else:
        ds = load_dataset(config.dataset_name, split=split)

    block = config.seq_len + 1
    texts = [ex[config.text_field] for ex in ds if len(ex[config.text_field].strip()) >= config.min_text_chars]
    if config.num_examples is not None:
        texts = texts[: config.num_examples]
    print(f"[pretrain] {len(texts)} non-trivial text rows after filtering (requested {config.num_examples})", flush=True)

    all_ids = [tokenizer(t, truncation=True, max_length=block, padding="max_length")["input_ids"] for t in texts]
    ids_t = torch.tensor(all_ids, dtype=torch.long)
    n_batches = ids_t.shape[0] // config.batch
    return ids_t[: n_batches * config.batch].view(n_batches, config.batch, block)


def run_pretrain_training(rank: int, signaling_url: str, config: PretrainConfig, job_id: str = "pretrain_pipeline") -> list[float]:
    from quic_dist.training_utils import set_seed, ExperimentLogger, CheckpointState, save_checkpoint, load_checkpoint

    set_seed(config.seed)

    local_gpu = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_gpu}")
    is_first = rank == 0
    is_last = rank == config.world_size - 1
    prev_rank = rank - 1 if rank > 0 else None
    next_rank = rank + 1 if rank < config.world_size - 1 else None

    logger = ExperimentLogger(config.log_path, rank)
    logger.log_config(config)

    quic_dist.init_process_group(
        signaling_url=signaling_url, rank=rank, world_size=config.world_size, job_id=job_id,
        timeout=timedelta(seconds=config.connect_timeout_s),
    )
    print(f"[rank {rank}] process group ready (local GPU {local_gpu})", flush=True)

    model, stage_layers, embed, norm, rotary, lm_head, hidden_size, trainable, tokenizer = build_pretrain_stage_model(
        rank, local_gpu, config
    )
    optimizer = torch.optim.AdamW(trainable, lr=config.lr)

    batches = build_pretrain_dataset(tokenizer, config)
    n_steps = batches.shape[0]
    total_steps = n_steps * config.epochs
    print(f"[rank {rank}] pretrain: {config.epochs} epochs x {n_steps} steps/epoch = {total_steps} steps "
          f"({sum(p.numel() for m in stage_layers for p in m.parameters())} local-layer params)", flush=True)

    resume_step = 0
    if config.checkpoint_dir:
        ckpt_state = load_checkpoint(config.checkpoint_dir, rank, model, optimizer, map_location=device)
        if ckpt_state is not None:
            resume_step = ckpt_state.step
            print(f"[rank {rank}] resumed from checkpoint at step {resume_step}/{total_steps}", flush=True)

    # Real bug found via finetune.py's kill-mid-run + resume test (see
    # that module's identical fix): reusing the SAME job_id for
    # _step_barrier's store keys across a crashed attempt and its resume
    # can silently under-satisfy the barrier. Namespacing by resume_step
    # guarantees a resumed attempt never touches a crashed attempt's
    # leftover keys.
    barrier_job_id = f"{job_id}_r{resume_step}"

    # Real barrier before any real tensor crosses the wire - see
    # rlhf.py's run_dpo_training's identical barrier for the real cross-
    # machine timeout this prevents (a fast-loading rank starting to
    # send before a slow-loading rank is even listening).
    dist.barrier()
    print(f"[rank {rank}] all ranks finished loading, starting training", flush=True)

    losses: list[float] = []
    t_start = time.monotonic()
    step_counter = 0

    for epoch in range(config.epochs):
        for b in range(n_steps):
            step_counter += 1
            if step_counter <= resume_step:
                continue  # already completed in a previous, interrupted run
            _step_barrier(signaling_url, config, f"pretrain_{epoch}_{b}", job_id=barrier_job_id)
            tag = step_counter % 8
            block = batches[b]
            optimizer.zero_grad()
            position_ids = torch.arange(config.seq_len, device=device).unsqueeze(0).expand(config.batch, -1)

            if is_first:
                input_ids = block[:, :-1].to(device)
                hidden = embed(input_ids).to(config.torch_dtype)
            else:
                recv_buf = torch.zeros(config.batch, config.seq_len, hidden_size, dtype=config.torch_dtype)
                dist.recv(recv_buf, src=prev_rank, tag=tag)
                hidden = recv_buf.to(device).requires_grad_(True)

            pos_emb = rotary(hidden, position_ids) if rotary is not None else None
            out = hidden
            for layer in stage_layers:
                out = run_decoder_layer(layer, out, attention_mask=None, position_ids=position_ids, position_embeddings=pos_emb)

            if is_last:
                labels = block[:, 1:].to(device)
                out = norm(out)
                logits = lm_head(out.to(lm_head.weight.dtype))
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1))
                loss.backward()
                losses.append(loss.item())
            else:
                dist.send(out.detach().to(config.torch_dtype).cpu(), dst=next_rank, tag=tag)
                grad = torch.zeros(config.batch, config.seq_len, hidden_size, dtype=config.torch_dtype)
                dist.recv(grad, src=next_rank, tag=tag)
                out.backward(grad.to(device))

            if not is_first:
                dist.send(hidden.grad.detach().to(config.torch_dtype).cpu(), dst=prev_rank, tag=tag)

            optimizer.step()
            if step_counter <= 3 or step_counter % config.log_every == 0:
                msg = f"[rank {rank}] step {step_counter}/{total_steps}"
                if is_last:
                    msg += f" loss={losses[-1]:.4f}"
                print(msg, flush=True)
            if is_last and losses:
                logger.log(event="step", step=step_counter, epoch=epoch, loss=losses[-1])

            if config.checkpoint_dir and config.checkpoint_every > 0 and step_counter % config.checkpoint_every == 0:
                path = save_checkpoint(
                    config.checkpoint_dir, rank, model, optimizer,
                    CheckpointState(step=step_counter, epoch=epoch, batch_index=b),
                    keep_last=config.checkpoint_keep_last,
                )
                print(f"[rank {rank}] checkpoint saved: {path}", flush=True)

        elapsed = time.monotonic() - t_start
        if is_last:
            print(f"[rank {rank}] epoch {epoch}/{config.epochs} loss={losses[-1]:.4f} elapsed={elapsed:.1f}s", flush=True)
        else:
            print(f"[rank {rank}] epoch {epoch}/{config.epochs} elapsed={elapsed:.1f}s", flush=True)

    total_elapsed = time.monotonic() - t_start
    print(f"[rank {rank}] pretrain training DONE in {total_elapsed:.1f}s ({total_steps} steps)", flush=True)
    if is_last and losses:
        print(f"[rank {rank}] loss avg first {min(8,len(losses))} steps: {sum(losses[:8])/min(8,len(losses)):.4f}", flush=True)
        print(f"[rank {rank}] loss avg last {min(8,len(losses))} steps:  {sum(losses[-8:])/min(8,len(losses)):.4f}", flush=True)

    dist.barrier()
    pg = dist.distributed_c10d._get_default_group()
    dist.destroy_process_group()
    pg.close_connections()
    print(f"[rank {rank}] ALL DONE", flush=True)
    return losses
