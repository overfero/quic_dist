"""Generic N-stage pipeline-parallel LoRA/QLoRA fine-tuning, config-driven.

Extracts the boilerplate that was duplicated across every
*_pipeline_rank.py causal-LM training script in examples/
(real_llm_pipeline_rank.py's Qwen2.5-0.5B, qwen38_27b_pipeline_rank.py's
Qwen3.8-27B hybrid-attention model) - a new model/dataset combination
needs a `PipelineConfig`, not a new copy of the training loop. What's
extracted, concretely:

- Per-rank `device_map` construction, including UNEVEN layer splits
  (some ranks carry extra fixed weight - a large lm_head/embed_tokens -
  that no middle rank pays for; qwen38_27b_pipeline_rank.py hit a real
  CUDA OOM from assuming an even split and had to rebalance by hand -
  `PipelineConfig.stage_layer_counts` makes that a config value, not a
  code change).
- LoRA + optional 4-bit/8-bit quantization via peft/bitsandbytes.
- The generic recv -> layers -> send forward/backward loop, for ANY
  world_size (not hardcoded to 2 or 4), with first/last/middle stages
  each handled once, not copy-pasted per script.
- quic_dist init/teardown, including the shutdown()-on-destroy path.

Model layer introspection is via DOTTED ATTRIBUTE PATHS
(`PipelineConfig.layers_attr` etc.), not a fixed assumed nesting -
different architectures put the decoder layers at different depths.
Confirmed via direct testing on two real, differently-shaped models:
`model.layers` works for a plain AutoModelForCausalLM (Qwen2.5-0.5B);
`model.model.layers` was needed for Qwen3_5ForCausalLM, whose config is
technically a VLM's, and whose `AutoModelForCausalLM.from_pretrained`
resolution auto-selects a text-only class one level deeper.

NOT covered, deliberately: non-text modalities. vision_pipeline_rank.py
trains real images (pixel tensors, not tokenized text) through a real
ViT - genuinely different data pipeline and forward signature, not
force-fit into this module. Keep vision/multimodal training as its own
script; use this module for text/causal-LM LoRA and QLoRA.

Usage: write a `PipelineConfig` (directly, or loaded from YAML/JSON via
`PipelineConfig.from_file`), then call `run_pipeline_training(rank,
signaling_url, config, job_id=...)` from a tiny per-model launcher
script - see examples/pipeline_finetune_rank.py and the example configs
under examples/configs/ for two real, validated examples (0.5B and 27B).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn as nn

import quic_dist
import torch.distributed as dist


@dataclass
class PipelineConfig:
    model_path: str
    world_size: int
    num_layers: int

    # Layer split across stages - None means as-even-as-possible. Set
    # explicitly when a boundary stage carries extra fixed weight (a
    # large lm_head/embed_tokens) that would otherwise tip it into OOM
    # first - see this module's docstring for why that's real, not
    # theoretical.
    stage_layer_counts: list[int] | None = None

    # Dotted attribute paths into the loaded model, resolved via
    # resolve_attr() below. Defaults match a plain AutoModelForCausalLM;
    # override layers_attr (etc.) for a model whose CausalLM class nests
    # the decoder one level deeper (e.g. "model.model.layers").
    layers_attr: str = "model.layers"
    embed_attr: str = "model.embed_tokens"
    norm_attr: str = "model.norm"
    rotary_attr: str | None = "model.rotary_emb"
    lm_head_attr: str = "lm_head"

    # LoRA (peft.LoraConfig passthrough)
    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_dropout: float = 0.0

    # Quantization - "4bit" (QLoRA), "8bit", or "none" (plain LoRA)
    quantization: str = "4bit"
    bnb_4bit_quant_type: str = "nf4"
    compute_dtype: str = "bfloat16"  # must match the base model's own native dtype - a real
                                      # mismatch (fp16 vs bf16) crashed a real run; see
                                      # qwen38_27b_pipeline_rank.py's history for the traceback.
    cpu_offload_unused_layers: bool = False  # True if a single GPU can't hold this
                                              # rank's whole slice even quantized - see
                                              # build_device_map()'s docstring

    # Peft/torchao environment workaround - see examples/real_llm_pipeline_rank.py's
    # module docstring for the real bug this works around in this project's
    # environment (peft probing an incompatible torchao unconditionally).
    patch_torchao_check: bool = True

    # Dataset - a real HF dataset, tokenized as plain causal-LM text.
    # For anything more structured than "one text field", write batches
    # yourself and call run_pipeline_training's lower-level pieces
    # directly instead (build_stage_model + the loop body) rather than
    # forcing a mismatched dataset spec through build_dataset().
    #
    # training_mode: "cpt" (default, unchanged from this module's
    # original behavior) trains on the WHOLE `text_field` blob as plain
    # next-token continuation, no masking - genuine continued
    # pre-training semantics (raw domain text, every token supervises
    # the loss). "sft" additionally masks the loss to ONLY the response
    # region (set `prompt_field`/`response_field` instead of
    # `text_field`) - real instruction-tuning semantics, mirroring
    # rlhf.py's DPO dataset's token-boundary approach (tokenize prompt
    # and response SEPARATELY then concatenate ids, so the boundary is
    # exact in token space, not string space).
    training_mode: str = "cpt"
    dataset_name: str = "tatsu-lab/alpaca"
    dataset_split: str = "train"
    num_examples: int | None = 64
    text_field: str = "text"
    prompt_field: str = "instruction"
    response_field: str = "output"

    # Training
    seq_len: int = 96
    batch: int = 1
    epochs: int = 3
    lr: float = 1e-4
    log_every: int = 16
    connect_timeout_s: int = 300

    @classmethod
    def from_file(cls, path: str) -> "PipelineConfig":
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


def resolve_attr(obj, dotted_path: str):
    for part in dotted_path.split("."):
        obj = getattr(obj, part)
    return obj


def run_decoder_layer(layer, hidden_states, **kwargs):
    """Calls one decoder layer and returns just the hidden-states
    tensor, version-robustly. Some transformers releases' decoder layer
    `forward` returns the hidden states directly; others (confirmed
    directly, not assumed: transformers==4.53.3's
    `Qwen2DecoderLayer.forward`) ALWAYS return a tuple
    `(hidden_states,)` (plus attention weights if output_attentions=True)
    regardless of use_cache/past_key_value - a real `AttributeError:
    'tuple' object has no attribute 'dtype'` crash inside the NEXT
    layer's input_layernorm is what surfaced this, in
    pipeline_generate()'s decode loop, the first code path exercised
    against a freshly-installed transformers after a full local
    environment reset. Every place in this project that loops over raw
    decoder-layer submodules (finetune.py's SFT loop, rlhf.py's DPO/
    RM/PRM/GRPO/PPO/RLOO forced-forward and pipeline_generate,
    multimodal.py's VLM loop) needs just the tensor - this is the one
    place that difference is absorbed, so a future transformers version
    changing this again only needs a fix here."""
    out = layer(hidden_states, **kwargs)
    return out[0] if isinstance(out, tuple) else out


def _step_barrier(signaling_url, config, tag: str, job_id: str = "", timeout_s: int = 300):
    """A REAL per-step barrier - unlike `dist.barrier()`, which is only
    safe to call ONCE per process group's lifetime. Found via a real
    cross-machine GRPO run (see quic_dist/rlhf.py's identical helper,
    which this mirrors): PyTorch's own `_store_based_barrier` keys its
    store entry ONLY by the process group's name, not by call site or
    call count, so a second `dist.barrier()` call in the same process
    group reads the already-satisfied key from the FIRST call and
    returns immediately - a silent no-op. Reimplements the same
    add-then-wait-for-last-worker pattern directly against the store,
    with a fresh key per `tag` so each call is genuinely independent.

    `job_id` is included in the key on top of `tag` - see rlhf.py's
    identical helper's docstring for the real hang this fixes (the
    signaling server's `/kv/*` store is shared across every run that
    ever points at it; two different runs reaching the same `tag`,
    trivial across repeated dev-loop restarts, silently share a
    counter, and an earlier crashed run's orphaned `add()` calls push
    the count past `world_size` forever since the check is exact
    equality, not `>=`)."""
    from quic_dist.store import QuicRendezvousStore

    store = QuicRendezvousStore(signaling_url, timeout=timedelta(seconds=timeout_s))
    key = f"step_barrier_{job_id}_{tag}"
    count = store.add(key, 1)
    if count == config.world_size:
        store.set(f"{key}:done", "1")
    store.wait([f"{key}:done"])


def stage_range(rank: int, config: PipelineConfig) -> range:
    counts = config.stage_layer_counts
    if counts is None:
        base = config.num_layers // config.world_size
        counts = [base] * config.world_size
        counts[-1] += config.num_layers - base * config.world_size
    if len(counts) != config.world_size or sum(counts) != config.num_layers:
        raise ValueError(
            f"stage_layer_counts {counts} must have world_size={config.world_size} "
            f"entries summing to num_layers={config.num_layers}"
        )
    start = sum(counts[:rank])
    return range(start, start + counts[rank])


def build_device_map(rank: int, local_gpu: int, config: PipelineConfig) -> dict:
    """Puts ONLY this rank's own layers on its GPU. The rest have to be
    pointed SOMEWHERE (a transformers device_map requirement) - "cpu" +
    cpu_offload_unused_layers=True (which sets
    llm_int8_enable_fp32_cpu_offload on the BitsAndBytesConfig) lands
    them on the meta device with zero real memory ever materialized -
    confirmed via a direct load test, not assumed. Only relevant when
    quantization != "none"; leave cpu_offload_unused_layers=False for
    plain LoRA, where every rank naturally only instantiates its own
    slice of the model in the first place."""
    my_layers = set(stage_range(rank, config))
    is_first = rank == 0
    is_last = rank == config.world_size - 1
    layers_prefix = config.layers_attr.replace("model.", "", 1) if config.layers_attr.startswith("model.") else config.layers_attr
    other = "cpu" if config.cpu_offload_unused_layers else local_gpu
    device_map = {}
    for i in range(config.num_layers):
        device_map[f"{config.layers_attr}.{i}"] = local_gpu if i in my_layers else other
    device_map[config.embed_attr] = local_gpu if is_first else other
    device_map[config.norm_attr] = local_gpu if is_last else other
    if config.rotary_attr:
        device_map[config.rotary_attr] = local_gpu
    device_map[config.lm_head_attr] = local_gpu if is_last else other
    return device_map


def build_stage_model(rank: int, local_gpu: int, config: PipelineConfig):
    """Returns (peft_model, stage_layers, embed_tokens, norm, rotary_emb_or_None, lm_head, hidden_size)."""
    if config.patch_torchao_check:
        import peft.tuners.lora.torchao as torchao_mod

        torchao_mod.is_torchao_available = lambda: False

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from transformers.utils import logging as hflog
    from peft import LoraConfig, get_peft_model

    hflog.disable_progress_bar()
    hflog.set_verbosity_error()

    device_map = build_device_map(rank, local_gpu, config)
    # Deliberately NOT passing dtype=config.torch_dtype on the quantized
    # paths below. bitsandbytes skips quantizing lm_head/embed_tokens by
    # design - forcing the WHOLE model's dtype would silently downcast
    # those two to config.torch_dtype too, defeating the reason they're
    # skipped (higher-precision output projection/embedding lookup). The
    # real fix for the dtype mismatch this originally hit (a bf16
    # checkpoint's lm_head fed an fp16 activation -> `float != BFloat16`
    # RuntimeError, found via a direct validation run) is instead in
    # run_pipeline_training(): the activation tensor gets cast to
    # embed_tokens'/lm_head's OWN weight dtype right at the point it
    # crosses into them, not the other way around.
    if config.quantization == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=config.torch_dtype,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            llm_int8_enable_fp32_cpu_offload=config.cpu_offload_unused_layers,
        )
        model = AutoModelForCausalLM.from_pretrained(config.model_path, quantization_config=bnb_cfg, device_map=device_map)
    elif config.quantization == "8bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=config.cpu_offload_unused_layers
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.model_path, quantization_config=bnb_cfg, device_map=device_map
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(config.model_path, dtype=config.torch_dtype).to(f"cuda:{local_gpu}")

    lora_cfg = LoraConfig(
        r=config.lora_r, lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules, lora_dropout=config.lora_dropout,
    )
    peft_model = get_peft_model(model, lora_cfg)

    layers = resolve_attr(peft_model.base_model.model, config.layers_attr)
    embed = resolve_attr(peft_model.base_model.model, config.embed_attr)
    norm = resolve_attr(peft_model.base_model.model, config.norm_attr)
    rotary = resolve_attr(peft_model.base_model.model, config.rotary_attr) if config.rotary_attr else None
    lm_head = resolve_attr(peft_model.base_model.model, config.lm_head_attr)
    my_range = stage_range(rank, config)
    stage_layers = layers[my_range.start : my_range.stop]

    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"[rank {rank}] trainable params: {n_trainable}, layers {list(my_range)}", flush=True)

    hidden_size = getattr(peft_model.base_model.model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = peft_model.base_model.model.config.text_config.hidden_size
    return peft_model, stage_layers, embed, norm, rotary, lm_head, hidden_size


def build_dataset(tokenizer, config: PipelineConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (input_ids, response_mask), each (n_batches, batch,
    block). In "cpt" mode response_mask is all-ones (every token
    supervises the loss, matching this function's original behavior
    exactly). In "sft" mode response_mask is 1 only over the response
    region - computed in TOKEN space (prompt and response tokenized
    SEPARATELY then concatenated), not by string-matching a rendered
    prompt+response, so the boundary is exact."""
    from datasets import load_dataset

    split = config.dataset_split if config.num_examples is None else f"{config.dataset_split}[:{config.num_examples}]"
    ds = load_dataset(config.dataset_name, split=split)
    block = config.seq_len + 1
    pad_id = tokenizer.pad_token_id

    all_ids = []
    all_mask = []
    if config.training_mode == "sft":
        for ex in ds:
            prompt_ids = tokenizer(ex[config.prompt_field], truncation=True, max_length=block, add_special_tokens=True)["input_ids"]
            response_ids = tokenizer(ex[config.response_field], truncation=True, max_length=block, add_special_tokens=False)["input_ids"]
            ids = (prompt_ids + response_ids)[:block]
            mask = ([0] * len(prompt_ids) + [1] * len(response_ids))[:block]
            pad = block - len(ids)
            if pad > 0:
                ids = ids + [pad_id] * pad
                mask = mask + [0] * pad
            all_ids.append(ids)
            all_mask.append(mask)
    else:
        for ex in ds:
            ids = tokenizer(ex[config.text_field], truncation=True, max_length=block, padding="max_length")["input_ids"]
            all_ids.append(ids)
            all_mask.append([1] * block)

    ids_t = torch.tensor(all_ids, dtype=torch.long)
    mask_t = torch.tensor(all_mask, dtype=torch.long)
    n_batches = ids_t.shape[0] // config.batch
    ids_t = ids_t[: n_batches * config.batch].view(n_batches, config.batch, block)
    mask_t = mask_t[: n_batches * config.batch].view(n_batches, config.batch, block)
    return ids_t, mask_t


def run_pipeline_training(rank: int, signaling_url: str, config: PipelineConfig, job_id: str = "pipeline_finetune", local_gpu: int | None = None) -> list[float]:
    """Runs the full N-stage pipeline training loop for this rank and
    returns the last-stage's per-step loss list (empty on non-last
    ranks). Blocks until training completes; the caller's script is
    just this call plus argument parsing - see
    examples/pipeline_finetune_rank.py."""
    if local_gpu is None:
        local_gpu = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_gpu}")
    is_first = rank == 0
    is_last = rank == config.world_size - 1
    prev_rank = rank - 1 if rank > 0 else None
    next_rank = rank + 1 if rank < config.world_size - 1 else None

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quic_dist.init_process_group(
        signaling_url=signaling_url, rank=rank, world_size=config.world_size, job_id=job_id,
        timeout=timedelta(seconds=config.connect_timeout_s),
    )
    print(f"[rank {rank}] process group ready (local GPU {local_gpu})", flush=True)

    peft_model, stage_layers, embed, norm, rotary, lm_head, hidden_size = build_stage_model(rank, local_gpu, config)

    # Real bug found via a cross-machine 27B DPO run (rlhf.py): model
    # loading takes very different real wall-clock time per rank (disk
    # speed, machine load) - without a barrier here, a fast-loading rank
    # can reach the training loop and start sending real tensors while a
    # slow-loading rank is still mid-load, and the underlying QUIC
    # connection's own idle timeout closes the connection before the
    # slow rank ever gets there.
    dist.barrier()
    print(f"[rank {rank}] all ranks finished loading, starting training", flush=True)

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr)

    batches, response_masks = build_dataset(tokenizer, config)
    n_steps = batches.shape[0]
    total_steps = n_steps * config.epochs
    print(f"[rank {rank}] {config.epochs} epochs x {n_steps} steps/epoch = {total_steps} steps", flush=True)

    losses: list[float] = []
    t_start = time.monotonic()
    step_counter = 0

    for epoch in range(config.epochs):
        for b in range(n_steps):
            # See rlhf.py's run_dpo_training's identical per-step
            # barrier for why: a rank finishing a step faster than
            # another can leave the slower rank's connection idle long
            # enough for the underlying QUIC connection's own idle
            # timeout to fire before the next step's first send/recv.
            # Uses _step_barrier, NOT dist.barrier() - see that
            # function's docstring for why a second dist.barrier() call
            # is a silent no-op.
            _step_barrier(signaling_url, config, f"sft_{epoch}_{b}", job_id=job_id)
            tag = step_counter % 8
            step_counter += 1
            block = batches[b]
            mask = response_masks[b]
            optimizer.zero_grad()
            position_ids = torch.arange(config.seq_len, device=device).unsqueeze(0).expand(config.batch, -1)

            if is_first:
                input_ids = block[:, :-1].to(device)
                # embed_tokens may be in its OWN dtype (bitsandbytes
                # leaves it unquantized/uncast by default) - cast its
                # output to config.torch_dtype here, once, rather than
                # forcing embed_tokens itself to a different precision
                # than the checkpoint intended.
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
                resp_mask = mask[:, 1:].to(device).float()
                out = norm(out)
                # Symmetric to the embed_tokens cast above: lm_head is
                # also left unquantized/uncast by bitsandbytes by
                # design, so cast the ACTIVATION to lm_head's own
                # weight dtype right here rather than forcing lm_head
                # itself into config.torch_dtype.
                logits = lm_head(out.to(lm_head.weight.dtype))
                # "cpt" mode's mask is all-ones, so this is exactly the
                # original unmasked mean cross-entropy - unchanged
                # behavior for every existing validated config. "sft"
                # mode's mask zeroes out prompt/pad positions, so only
                # response tokens contribute - a real masked mean, not
                # just zeroing the loss AFTER an unmasked mean (that
                # would still be wrong-denominator).
                per_token = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1), reduction="none"
                ).reshape(resp_mask.shape)
                loss = (per_token * resp_mask).sum() / resp_mask.sum().clamp(min=1)
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
                print(f"[rank {rank}] step {step_counter}/{total_steps}", flush=True)

        elapsed = time.monotonic() - t_start
        if is_last:
            print(f"[rank {rank}] epoch {epoch}/{config.epochs} loss={losses[-1]:.4f} elapsed={elapsed:.1f}s", flush=True)
        else:
            print(f"[rank {rank}] epoch {epoch}/{config.epochs} elapsed={elapsed:.1f}s", flush=True)

    total_elapsed = time.monotonic() - t_start
    print(f"[rank {rank}] training DONE in {total_elapsed:.1f}s ({total_steps} steps)", flush=True)
    if is_last and losses:
        print(f"[rank {rank}] loss avg first epoch: {sum(losses[:n_steps]) / n_steps:.4f}", flush=True)
        print(f"[rank {rank}] loss avg last epoch:  {sum(losses[-n_steps:]) / n_steps:.4f}", flush=True)

    dist.barrier()
    pg = dist.distributed_c10d._get_default_group()
    dist.destroy_process_group()
    pg.close_connections()
    print(f"[rank {rank}] ALL DONE", flush=True)
    return losses
