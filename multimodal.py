"""Config-driven pipeline-parallel multimodal (vision-language) SFT,
sibling to finetune.py (text SFT/CPT) and rlhf.py (DPO/GRPO/PPO/RM/PRM).

finetune.py's own docstring deliberately excludes non-text modalities -
a real image+text forward pass needs a vision tower run BEFORE any
decoder layer, which no text-only config field can express. This module
is that exclusion's actual home, not a workaround: same dotted-attribute-
path philosophy (`MultimodalConfig.layers_attr` etc.), same LoRA/
quantization fields (`RLHFModelConfig`-compatible, reused by name, not
copied), same barrier discipline (`dist.barrier()` once after loading,
`_step_barrier` every step - see rlhf.py's `_step_barrier` docstring for
the real bug a plain second `dist.barrier()` call silently reproduces).

Concretely new here, not present in finetune.py's text pipeline:

- `build_multimodal_stage_model`: loads via
  `AutoModelForImageTextToText` (LlavaForConditionalGeneration is NOT
  registered under `AutoModelForCausalLM` - confirmed via direct
  `AutoModelForImageTextToText._model_mapping` inspection, not assumed),
  and additionally places the vision tower + multimodal projector on
  rank 0's device_map (they only ever run there - see below).
- Stage 0's forward step additionally runs the REAL vision tower +
  projector (`peft_model.base_model.model.model.get_image_features` -
  the model's own method, not reimplemented) and merges the resulting
  image embeddings into the text embedding sequence via
  `masked_scatter` at `image_token_id` positions - the EXACT merge
  logic `LlavaModel.forward` itself uses internally, confirmed by
  reading its source directly (transformers 4.53.3) rather than
  guessed. Every stage AFTER rank 0 needs no changes at all: once
  merged, `inputs_embeds` is just a normal `(B, T, hidden)` activation
  tensor flowing through plain Qwen2 decoder layers with plain 1D
  `position_ids` - LLaVA-style models use NO special multimodal
  position encoding (unlike e.g. Qwen2-VL's M-RoPE, which needs
  per-image grid metadata threaded through every stage - deliberately
  picked a LLaVA-family model for the first real proof to avoid that
  extra complexity; a future M-RoPE-based VLM would need
  pipeline_generate()-style metadata broadcast, not a config field).
- `build_vlm_sft_dataset`: one real (image, prompt, response) example
  per pipeline step, ragged (not batched) - mirrors rlhf.py's PRM
  dataset for the same reason (variable prompt+response token length
  per example - real images can produce different total sequence
  lengths even when the image itself is resized to a fixed shape,
  since the TEXT portion varies). Response masking uses the same
  token-space boundary approach as finetune.py's SFT mode and rlhf.py's
  DPO dataset (prompt+image tokenized/processed first, response
  tokenized separately, concatenated - exact boundary, no string
  matching).

NOT covered here either: models needing per-example multimodal position
metadata (M-RoPE), video, audio, or more than one image per example.
Real, validated scope: single-image visual instruction tuning on a
LLaVA-family model, LoRA/QLoRA, any world_size.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import quic_dist
import torch.distributed as dist

from quic_dist.finetune import resolve_attr, stage_range, build_device_map, run_decoder_layer
from quic_dist.rlhf import _step_barrier, _teardown


@dataclass
class MultimodalConfig:
    """Field-name-compatible with RLHFModelConfig for the model/LoRA/
    quant side (see that class's own docstring for why that matters),
    but with LLaVA-family nesting as the DEFAULT (not a plain CausalLM's)
    since that's this module's only real, validated target - override
    per-model via direct empirical inspection, same as every other
    config in this repo, never copied from another model's memory."""

    model_path: str
    world_size: int
    num_layers: int
    stage_layer_counts: list[int] | None = None

    layers_attr: str = "model.language_model.layers"
    embed_attr: str = "model.language_model.embed_tokens"
    norm_attr: str = "model.language_model.norm"
    rotary_attr: str | None = "model.language_model.rotary_emb"
    lm_head_attr: str = "lm_head"
    vision_tower_attr: str = "model.vision_tower"
    projector_attr: str = "model.multi_modal_projector"
    image_token_id: int = 151646  # llava-hf/llava-interleave-qwen-0.5b-hf's real value -
                                   # confirmed via AutoConfig, not a general LLaVA constant;
                                   # a different checkpoint's config.image_token_index may differ.

    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_dropout: float = 0.0

    quantization: str = "none"  # this module's one real proof target (0.5B text + SigLIP
                                 # vision tower) fits a T4 unquantized - "4bit"/"8bit" still
                                 # work (bitsandbytes quantizes the vision tower's linear
                                 # layers too, since target_modules-independent quantization_config
                                 # applies to every nn.Linear the device_map places on a real GPU),
                                 # just not needed at this scale.
    bnb_4bit_quant_type: str = "nf4"
    compute_dtype: str = "bfloat16"
    cpu_offload_unused_layers: bool = False
    patch_torchao_check: bool = True

    connect_timeout_s: int = 300
    log_every: int = 4
    lr: float = 1e-4

    # HuggingFaceH4/llava-instruct-mix-vsft's real schema (confirmed by
    # loading 2 real rows directly, not assumed from its name): a
    # `messages` field - a chat-turn list, each turn's `content` a list
    # of {"type":"text","text":...} / {"type":"image","index":N} parts -
    # plus a top-level `images` field (list of PIL images). Real examples
    # often have multiple turns about the same image; this module trains
    # on the FIRST user/assistant exchange only (real, correct, just a
    # smaller slice of each example's signal - multi-turn packing is a
    # genuinely separate feature, not attempted here).
    dataset_name: str = "HuggingFaceH4/llava-instruct-mix-vsft"
    dataset_split: str = "train"
    num_examples: int | None = 32
    image_field: str = "images"
    messages_field: str = "messages"
    max_len: int = 1024  # prompt (incl. image placeholder tokens) + response token budget

    batch: int = 1  # one real example per pipeline step - see module docstring
    epochs: int = 3

    @classmethod
    def from_file(cls, path: str) -> "MultimodalConfig":
        text = Path(path).read_text()
        if path.endswith((".yaml", ".yml")):
            import yaml

            data = yaml.safe_load(text)
        else:
            import json

            data = json.loads(text)
        return cls(**data)

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.compute_dtype)


def build_multimodal_device_map(rank: int, local_gpu: int, config: MultimodalConfig) -> dict:
    """finetune.build_device_map() plus the vision tower + projector -
    they only ever run on rank 0 (see this module's docstring), so are
    placed there and nowhere else; every other rank sends them to
    "cpu"/local_gpu the same way it does for unused decoder layers."""
    device_map = build_device_map(rank, local_gpu, config)
    is_first = rank == 0
    other = "cpu" if config.cpu_offload_unused_layers else local_gpu
    device_map[config.vision_tower_attr] = local_gpu if is_first else other
    device_map[config.projector_attr] = local_gpu if is_first else other
    return device_map


def build_multimodal_stage_model(rank: int, local_gpu: int, config: MultimodalConfig):
    """Returns (peft_model, stage_layers, embed_tokens, norm,
    rotary_emb_or_None, lm_head, hidden_size, vision_tower_or_None,
    get_image_features_fn_or_None) - the last two are only non-None on
    rank 0 (is_first), matching where they're actually placed and used."""
    if config.patch_torchao_check:
        import peft.tuners.lora.torchao as torchao_mod

        torchao_mod.is_torchao_available = lambda: False

    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
    from transformers.utils import logging as hflog
    from peft import LoraConfig, get_peft_model

    hflog.disable_progress_bar()
    hflog.set_verbosity_error()

    device_map = build_multimodal_device_map(rank, local_gpu, config)
    if config.quantization == "4bit":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=config.torch_dtype,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            llm_int8_enable_fp32_cpu_offload=config.cpu_offload_unused_layers,
        )
        model = AutoModelForImageTextToText.from_pretrained(config.model_path, quantization_config=bnb_cfg, device_map=device_map)
    elif config.quantization == "8bit":
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=config.cpu_offload_unused_layers)
        model = AutoModelForImageTextToText.from_pretrained(config.model_path, quantization_config=bnb_cfg, device_map=device_map)
    else:
        # `dtype=` (the newer kwarg name) errors here specifically -
        # `LlavaForConditionalGeneration.__init__() got an unexpected
        # keyword argument 'dtype'` - a real, direct crash confirmed via
        # a live run, not anticipated. `torch_dtype=` (the older name)
        # works identically for this model/transformers version
        # combination; finetune.py's own unquantized branch uses
        # `dtype=` successfully because it does NOT also pass
        # device_map= in the same call - some code path specific to
        # device_map+multimodal model classes here doesn't accept the
        # newer kwarg name.
        model = AutoModelForImageTextToText.from_pretrained(config.model_path, torch_dtype=config.torch_dtype, device_map=device_map)

    lora_cfg = LoraConfig(
        r=config.lora_r, lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules, lora_dropout=config.lora_dropout,
    )
    peft_model = get_peft_model(model, lora_cfg)
    base = peft_model.base_model.model

    layers = resolve_attr(base, config.layers_attr)
    embed = resolve_attr(base, config.embed_attr)
    norm = resolve_attr(base, config.norm_attr)
    rotary = resolve_attr(base, config.rotary_attr) if config.rotary_attr else None
    lm_head = resolve_attr(base, config.lm_head_attr)
    my_range = stage_range(rank, config)
    stage_layers = layers[my_range.start: my_range.stop]

    is_first = rank == 0
    vision_tower = None
    get_image_features = None
    if is_first:
        vision_tower = resolve_attr(base, config.vision_tower_attr)
        # base.model is the raw LlavaModel - get_image_features is ITS
        # method (runs vision_tower + projector together), not
        # reimplemented here. Bound so callers don't need `base` at all.
        get_image_features = base.model.get_image_features

    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"[rank {rank}] trainable params: {n_trainable}, layers {list(my_range)}", flush=True)

    hidden_size = base.config.text_config.hidden_size
    return peft_model, stage_layers, embed, norm, rotary, lm_head, hidden_size, vision_tower, get_image_features


def _first_exchange(messages: list[dict]) -> tuple[str | None, str | None]:
    """messages: HuggingFaceH4/llava-instruct-mix-vsft's real per-turn
    format - [{"role": "user"|"assistant", "content": [{"type":"text",
    "text":...} | {"type":"image","index":N}, ...]}, ...] (confirmed by
    loading real rows directly). Returns (prompt_text, response_text)
    for the FIRST user turn immediately followed by an assistant turn -
    text parts concatenated in order, image parts replaced by a literal
    "<image>" placeholder (the processor's own expected token - see
    build_vlm_sft_dataset). Returns (None, None) if no such pair exists
    (malformed/assistant-first example - not assumed to happen, but not
    fatal if it does)."""

    def _render(content) -> str:
        parts = []
        for part in content:
            if part.get("type") == "image":
                parts.append("<image>")
            elif part.get("text"):
                parts.append(part["text"])
        return "".join(parts)

    for i, msg in enumerate(messages):
        if msg["role"] == "user" and i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
            return _render(msg["content"]), _render(messages[i + 1]["content"])
    return None, None


def build_vlm_sft_dataset(processor, tokenizer, config: MultimodalConfig) -> list[dict]:
    """Returns a flat list of {input_ids: (T,), pixel_values: (1,C,H,W),
    response_mask: (T,)} - kept ragged, one real example per pipeline
    step, batch collation deliberately not attempted (see module
    docstring). response_mask is 1 only over the response region -
    prompt (incl. image placeholder tokens) and response are
    tokenized/processed SEPARATELY then concatenated, so the boundary
    is exact in token space, matching every other dataset builder in
    this repo (finetune.py's SFT mode, rlhf.py's DPO dataset)."""
    from datasets import load_dataset

    split = config.dataset_split if config.num_examples is None else f"{config.dataset_split}[:{config.num_examples}]"
    ds = load_dataset(config.dataset_name, split=split)

    examples = []
    for ex in ds:
        image = ex[config.image_field]
        if isinstance(image, list):
            image = image[0]
        # Real, not hypothetical: some real examples in this dataset are
        # RGBA (PNG) or other non-3-channel modes - the SigLIP image
        # processor's channel-dimension inference hard-crashes
        # ("Unable to infer channel dimension format") on anything that
        # isn't standard RGB, found via a live run (index 10 of the
        # first 16 real rows is RGBA), not anticipated.
        image = image.convert("RGB")

        prompt_text, response_text = _first_exchange(ex[config.messages_field])
        if prompt_text is None:
            continue
        if "<image>" not in prompt_text:
            prompt_text = "<image>\n" + prompt_text

        proc_out = processor(text=prompt_text, images=image, return_tensors="pt")
        prompt_ids = proc_out["input_ids"][0]
        pixel_values = proc_out["pixel_values"]

        response_ids = tokenizer(response_text, add_special_tokens=False, truncation=True,
                                  max_length=config.max_len)["input_ids"]
        if len(prompt_ids) + len(response_ids) > config.max_len:
            continue  # real image token counts vary by resize grid - skip rather than
                      # truncate mid-image-placeholder-run, which would desync the
                      # merge's exact-count check in build_multimodal_stage_model's forward
        ids = torch.cat([prompt_ids, torch.tensor(response_ids, dtype=torch.long)])
        mask = torch.cat([torch.zeros(len(prompt_ids)), torch.ones(len(response_ids))])
        examples.append({"input_ids": ids, "pixel_values": pixel_values, "response_mask": mask})
    return examples


def run_multimodal_training(rank: int, signaling_url: str, config: MultimodalConfig,
                             job_id: str = "multimodal_pipeline") -> list[float]:
    from transformers import AutoTokenizer, AutoProcessor

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processor = AutoProcessor.from_pretrained(config.model_path)

    local_gpu = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_gpu}")
    is_first = rank == 0
    is_last = rank == config.world_size - 1
    prev_rank = rank - 1 if rank > 0 else None
    next_rank = rank + 1 if rank < config.world_size - 1 else None

    quic_dist.init_process_group(
        signaling_url=signaling_url, rank=rank, world_size=config.world_size, job_id=job_id,
        timeout=timedelta(seconds=config.connect_timeout_s),
    )
    print(f"[rank {rank}] process group ready (local GPU {local_gpu})", flush=True)

    (peft_model, stage_layers, embed, norm, rotary, lm_head, hidden_size,
     vision_tower, get_image_features) = build_multimodal_stage_model(rank, local_gpu, config)

    dist.barrier()
    print(f"[rank {rank}] all ranks finished loading, starting training", flush=True)

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr)

    # Only rank 0 needs the real dataset (it's the only rank that ever
    # embeds input_ids/runs the vision tower); other ranks still build
    # it (cheap - no model forward involved) purely to agree on
    # n_steps/total_steps for logging, same redundant-but-network-free
    # pattern build_dpo_dataset/build_prm_dataset already rely on.
    examples = build_vlm_sft_dataset(processor, tokenizer, config)
    n_steps = len(examples)
    total_steps = n_steps * config.epochs
    print(f"[rank {rank}] multimodal SFT: {config.epochs} epochs x {n_steps} examples = {total_steps} steps", flush=True)

    losses: list[float] = []
    t_start = time.monotonic()
    step_counter = 0

    for epoch in range(config.epochs):
        for ex in examples:
            _step_barrier(signaling_url, config, f"vlm_{step_counter}", job_id=job_id)
            step_counter += 1
            tag = step_counter % 8
            input_ids = ex["input_ids"].unsqueeze(0)  # (1, T)
            resp_mask = ex["response_mask"].to(device)
            T = input_ids.shape[1]
            optimizer.zero_grad()

            position_ids = torch.arange(T, device=device).unsqueeze(0)
            if is_first:
                inputs_embeds = embed(input_ids.to(device)).to(config.torch_dtype)
                pixel_values = ex["pixel_values"].to(device=device, dtype=config.torch_dtype)
                image_features = get_image_features(pixel_values=pixel_values)
                image_features = torch.cat(list(image_features), dim=0).to(inputs_embeds.dtype)
                special_image_mask = (input_ids.to(device) == config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                if inputs_embeds[special_image_mask].numel() != image_features.numel():
                    raise ValueError(
                        f"[rank {rank}] image token/feature count mismatch: "
                        f"{inputs_embeds[special_image_mask].numel()} vs {image_features.numel()} "
                        f"- see build_vlm_sft_dataset's max_len skip for why this shouldn't happen"
                    )
                hidden = inputs_embeds.masked_scatter(special_image_mask, image_features)
                hidden_in = None
            else:
                recv_buf = torch.zeros(1, T, hidden_size, dtype=config.torch_dtype)
                dist.recv(recv_buf, src=prev_rank, tag=tag)
                hidden_in = recv_buf.to(device).requires_grad_(True)
                hidden = hidden_in

            pos_emb = rotary(hidden, position_ids) if rotary is not None else None
            out = hidden
            for layer in stage_layers:
                out = run_decoder_layer(layer, out, attention_mask=None, position_ids=position_ids, position_embeddings=pos_emb)
                # Real, version-dependent difference found via a live run:
                # this transformers version's Qwen2DecoderLayer.forward
                # returns tuple[hidden_states, Optional[...]] (confirmed
                # via direct signature inspection - `-> tuple[...]`), NOT
                # a plain tensor like the newer transformers>=5.15 used
                # elsewhere in this repo (finetune.py/rlhf.py's own
                # Qwen2/Qwen3.8 pipelines - confirmed `-> torch.Tensor`
                # there). Unpack defensively rather than pin a
                # transformers version, since this repo already runs
                # against more than one.
                if isinstance(out, tuple):
                    out = out[0]

            if is_last:
                labels = input_ids[0, 1:].to(device)
                mask = resp_mask[1:].float()
                normed = norm(out)
                logits = lm_head(normed.to(lm_head.weight.dtype))[0, :-1, :]
                per_token = F.cross_entropy(logits.float(), labels, reduction="none")
                loss = (per_token * mask).sum() / mask.sum().clamp(min=1)
                loss.backward()
                losses.append(loss.item())
            else:
                dist.send(out.detach().to(config.torch_dtype).cpu(), dst=next_rank, tag=tag)
                grad = torch.zeros(1, T, hidden_size, dtype=config.torch_dtype)
                dist.recv(grad, src=next_rank, tag=tag)
                out.backward(grad.to(device))

            if not is_first:
                dist.send(hidden_in.grad.detach().to(config.torch_dtype).cpu(), dst=prev_rank, tag=tag)

            optimizer.step()
            if step_counter <= 3 or step_counter % config.log_every == 0:
                msg = f"[rank {rank}] step {step_counter}/{total_steps}"
                if is_last:
                    msg += f" vlm_sft_loss={losses[-1]:.4f}"
                print(msg, flush=True)

        elapsed = time.monotonic() - t_start
        if is_last:
            print(f"[rank {rank}] epoch {epoch}/{config.epochs} loss={losses[-1]:.4f} elapsed={elapsed:.1f}s", flush=True)
        else:
            print(f"[rank {rank}] epoch {epoch}/{config.epochs} elapsed={elapsed:.1f}s", flush=True)

    total_elapsed = time.monotonic() - t_start
    print(f"[rank {rank}] multimodal SFT training DONE in {total_elapsed:.1f}s ({total_steps} steps)", flush=True)
    if is_last and losses:
        print(f"[rank {rank}] loss avg first {min(8,len(losses))} steps: {sum(losses[:8])/min(8,len(losses)):.4f}", flush=True)
        print(f"[rank {rank}] loss avg last {min(8,len(losses))} steps:  {sum(losses[-8:])/min(8,len(losses)):.4f}", flush=True)

    _teardown(rank)
    return losses
