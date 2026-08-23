"""Config-driven pipeline-parallel DPO / GRPO / PPO over quic_dist,
sibling to finetune.py (which covers plain SFT LoRA/QLoRA only).

Reuses finetune.py's model-loading machinery UNCHANGED - stage_range(),
build_device_map(), and (most importantly) build_stage_model() are
called directly against the config objects below, not reimplemented.
This works because RLHFModelConfig below declares the exact same field
names finetune.PipelineConfig does for the model/LoRA/quant side; those
functions only ever access config by attribute name, so this is real
reuse (duck typing), not a copy. Do not rename a field on one side
without renaming it on the other.

Three training modes, increasing in what they need beyond SFT:

- DPO: needs a REFERENCE model's log-probs in addition to the policy's.
  Gotten for free via peft's `model.disable_adapter()` context manager -
  the frozen base weights ARE the reference model, no second model ever
  loaded. Two forward passes per step (policy w/ grad, reference w/o),
  no generation involved - closest to plain SFT.
- GRPO: needs on-policy SAMPLES. Since the model itself is split across
  ranks (no single rank holds enough of it to call .generate()),
  generation is done as an explicit token-by-token PIPELINE decode loop
  (pipeline_generate() below) - each new token is one full forward round
  trip through every stage, with each stage keeping its own local KV
  cache across steps (past_key_values threaded through consecutive
  calls). No value/critic model - advantage is the reward normalized
  within each prompt's sample group.
- PPO: GRPO's generation loop plus a learned value head (attached at the
  last stage only, next to lm_head) and GAE advantages instead of a
  group-normalized reward.

All three deliberately reuse pipeline_generate() and
pipeline_forward_backward() (the single-stage forward/backward-through-
this-rank's-layers step, extracted from finetune.run_pipeline_training's
loop body) rather than each reimplementing the recv/layers/send
plumbing.
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

from quic_dist.finetune import resolve_attr, stage_range, build_device_map, build_stage_model


@dataclass
class RLHFModelConfig:
    """Model/LoRA/quantization fields - field-name-identical to
    finetune.PipelineConfig's own, so finetune.build_stage_model() (and
    the stage_range/build_device_map it calls) work unmodified against
    this class too. See this module's docstring."""

    model_path: str
    world_size: int
    num_layers: int
    stage_layer_counts: list[int] | None = None

    layers_attr: str = "model.layers"
    embed_attr: str = "model.embed_tokens"
    norm_attr: str = "model.norm"
    rotary_attr: str | None = "model.rotary_emb"
    lm_head_attr: str = "lm_head"

    lora_r: int = 8
    lora_alpha: int = 16
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    lora_dropout: float = 0.0

    quantization: str = "4bit"
    bnb_4bit_quant_type: str = "nf4"
    compute_dtype: str = "bfloat16"
    cpu_offload_unused_layers: bool = False
    patch_torchao_check: bool = True

    connect_timeout_s: int = 300
    log_every: int = 8
    lr: float = 1e-4

    @classmethod
    def _load_raw(cls, path: str) -> dict:
        text = Path(path).read_text()
        if path.endswith((".yaml", ".yml")):
            import yaml

            return yaml.safe_load(text)
        import json

        return json.loads(text)

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.compute_dtype)


def _init_rank(rank, signaling_url, config, job_id, dtype_check=True):
    if dtype_check:
        pass  # compute_dtype vs model native dtype mismatch is the #1 real
              # crash seen in this repo (see finetune.py's history) -
              # nothing to check statically, just documenting the known
              # failure mode here so it's not re-discovered per mode.
    local_gpu = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{local_gpu}")
    quic_dist.init_process_group(
        signaling_url=signaling_url, rank=rank, world_size=config.world_size, job_id=job_id,
        timeout=timedelta(seconds=config.connect_timeout_s),
    )
    print(f"[rank {rank}] process group ready (local GPU {local_gpu})", flush=True)
    peft_model, stage_layers, embed, norm, rotary, lm_head, hidden_size = build_stage_model(rank, local_gpu, config)
    return local_gpu, device, peft_model, stage_layers, embed, norm, rotary, lm_head, hidden_size


def _teardown(rank):
    dist.barrier()
    pg = dist.distributed_c10d._get_default_group()
    dist.destroy_process_group()
    pg.close_connections()
    print(f"[rank {rank}] ALL DONE", flush=True)


# ---------------------------------------------------------------------------
# DPO
# ---------------------------------------------------------------------------


@dataclass
class DPOConfig(RLHFModelConfig):
    beta: float = 0.1

    dataset_name: str = "Intel/orca_dpo_pairs"
    dataset_split: str = "train"
    num_examples: int | None = 64
    prompt_field: str = "question"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"

    max_prompt_len: int = 64
    max_response_len: int = 64

    batch: int = 1  # pairs per step; pipeline batch is 2x this (chosen+rejected concatenated)
    epochs: int = 1

    @classmethod
    def from_file(cls, path: str) -> "DPOConfig":
        return cls(**cls._load_raw(path))

    @property
    def seq_len(self) -> int:
        return self.max_prompt_len + self.max_response_len


def build_dpo_dataset(tokenizer, config: DPOConfig):
    """Returns (input_ids, response_mask), each (n_batches, 2*batch,
    block). response_mask is 1 exactly over the completion tokens (not
    the prompt, not padding) - computed in TOKEN space by tokenizing
    prompt and completion separately and concatenating ids, not by
    string-matching a rendered prompt+completion, so the boundary is
    exact even when tokenization isn't whitespace-stable."""
    from datasets import load_dataset

    split = config.dataset_split if config.num_examples is None else f"{config.dataset_split}[:{config.num_examples}]"
    ds = load_dataset(config.dataset_name, split=split)
    block = config.seq_len
    pad_id = tokenizer.pad_token_id

    def make_example(prompt_text, resp_text):
        prompt_ids = tokenizer(prompt_text, truncation=True, max_length=config.max_prompt_len, add_special_tokens=True)["input_ids"]
        resp_ids = tokenizer(resp_text, truncation=True, max_length=config.max_response_len, add_special_tokens=False)["input_ids"]
        ids = prompt_ids + resp_ids
        mask = [0] * len(prompt_ids) + [1] * len(resp_ids)
        pad = block - len(ids)
        if pad > 0:
            ids = ids + [pad_id] * pad
            mask = mask + [0] * pad
        else:
            ids = ids[:block]
            mask = mask[:block]
        return ids, mask

    chosen_ids, chosen_mask, rejected_ids, rejected_mask = [], [], [], []
    for ex in ds:
        prompt = ex[config.prompt_field]
        c_ids, c_mask = make_example(prompt, ex[config.chosen_field])
        r_ids, r_mask = make_example(prompt, ex[config.rejected_field])
        chosen_ids.append(c_ids); chosen_mask.append(c_mask)
        rejected_ids.append(r_ids); rejected_mask.append(r_mask)

    chosen_ids = torch.tensor(chosen_ids, dtype=torch.long)
    rejected_ids = torch.tensor(rejected_ids, dtype=torch.long)
    chosen_mask = torch.tensor(chosen_mask, dtype=torch.long)
    rejected_mask = torch.tensor(rejected_mask, dtype=torch.long)

    n_batches = chosen_ids.shape[0] // config.batch
    chosen_ids = chosen_ids[: n_batches * config.batch].view(n_batches, config.batch, block)
    rejected_ids = rejected_ids[: n_batches * config.batch].view(n_batches, config.batch, block)
    chosen_mask = chosen_mask[: n_batches * config.batch].view(n_batches, config.batch, block)
    rejected_mask = rejected_mask[: n_batches * config.batch].view(n_batches, config.batch, block)

    input_ids = torch.cat([chosen_ids, rejected_ids], dim=1)       # (n_batches, 2*batch, block)
    response_mask = torch.cat([chosen_mask, rejected_mask], dim=1)  # (n_batches, 2*batch, block)
    return input_ids, response_mask


def _sequence_logp(logits, labels, mask):
    """Per-sequence sum of log p(label token) over masked positions.
    logits: (B, T, V) already at the predicted-token offset (i.e.
    logits[t] predicts labels[t]). mask/labels: (B, T)."""
    logprobs = F.log_softmax(logits.float(), dim=-1)
    token_logp = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(-1)


def run_dpo_training(rank: int, signaling_url: str, config: DPOConfig, job_id: str = "dpo_pipeline") -> list[float]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    local_gpu, device, peft_model, stage_layers, embed, norm, rotary, lm_head, hidden_size = _init_rank(
        rank, signaling_url, config, job_id
    )
    is_first = rank == 0
    is_last = rank == config.world_size - 1
    prev_rank = rank - 1 if rank > 0 else None
    next_rank = rank + 1 if rank < config.world_size - 1 else None

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr)

    input_ids_all, response_mask_all = build_dpo_dataset(tokenizer, config)
    n_steps = input_ids_all.shape[0]
    total_steps = n_steps * config.epochs
    pbatch = 2 * config.batch  # chosen+rejected concatenated
    seq_len = config.seq_len - 1  # predicting next-token, same -1 convention as finetune.py
    print(f"[rank {rank}] DPO: {config.epochs} epochs x {n_steps} steps/epoch = {total_steps} steps, "
          f"pipeline batch {pbatch} (={config.batch} pairs x2)", flush=True)

    losses: list[float] = []
    t_start = time.monotonic()
    step_counter = 0

    def forward_only(input_ids_or_None, tag, no_grad: bool):
        """One full local-stage forward. Returns (out_tensor,
        requires-grad-input-tensor-or-None) - the latter is what a
        later backward() call needs `.grad` from."""
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(pbatch, -1)
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            if is_first:
                hidden = embed(input_ids_or_None[:, :-1].to(device)).to(config.torch_dtype)
                hidden_in = None
            else:
                recv_buf = torch.zeros(pbatch, seq_len, hidden_size, dtype=config.torch_dtype)
                dist.recv(recv_buf, src=prev_rank, tag=tag)
                hidden_in = recv_buf.to(device)
                if not no_grad:
                    hidden_in = hidden_in.requires_grad_(True)
                hidden = hidden_in

            pos_emb = rotary(hidden, position_ids) if rotary is not None else None
            out = hidden
            for layer in stage_layers:
                out = layer(out, attention_mask=None, position_ids=position_ids, position_embeddings=pos_emb)

            if is_last:
                out = norm(out)
                logits = lm_head(out.to(lm_head.weight.dtype))
                return logits, hidden_in
            else:
                dist.send(out.detach().to(config.torch_dtype).cpu(), dst=next_rank, tag=tag)
                return out, hidden_in

    for epoch in range(config.epochs):
        for b in range(n_steps):
            step_counter += 1
            tag_ref = (step_counter % 4) * 2
            tag_policy = tag_ref + 1
            block = input_ids_all[b]           # (pbatch, block)
            mask = response_mask_all[b].to(device)  # (pbatch, block)
            optimizer.zero_grad()

            # --- reference pass: no grad, adapters disabled. Forward-only
            # round trip (every rank sends forward, nobody sends a
            # gradient back) - naturally deadlock-free since it never
            # waits on a reverse message.
            with peft_model.disable_adapter():
                ref_out, _ = forward_only(block, tag_ref, no_grad=True)

            # --- policy pass: grad-tracked, adapters enabled.
            policy_out, hidden_in = forward_only(block, tag_policy, no_grad=False)

            if is_last:
                labels = block[:, 1:].to(device)
                resp_mask = mask[:, 1:].float()
                ref_logp = _sequence_logp(ref_out, labels, resp_mask)
                policy_logp = _sequence_logp(policy_out, labels, resp_mask)

                B = config.batch
                pol_chosen, pol_rejected = policy_logp[:B], policy_logp[B:]
                ref_chosen, ref_rejected = ref_logp[:B], ref_logp[B:]
                logits_dpo = config.beta * ((pol_chosen - ref_chosen) - (pol_rejected - ref_rejected))
                loss = -F.logsigmoid(logits_dpo).mean()
                loss.backward()
                losses.append(loss.item())
            else:
                grad = torch.zeros(pbatch, seq_len, hidden_size, dtype=config.torch_dtype)
                dist.recv(grad, src=next_rank, tag=tag_policy)
                policy_out.backward(grad.to(device))

            if not is_first:
                dist.send(hidden_in.grad.detach().to(config.torch_dtype).cpu(), dst=prev_rank, tag=tag_policy)

            optimizer.step()
            if step_counter <= 3 or step_counter % config.log_every == 0:
                msg = f"[rank {rank}] step {step_counter}/{total_steps}"
                if is_last:
                    msg += f" dpo_loss={losses[-1]:.4f} margin={(pol_chosen-pol_rejected).mean().item():.3f}"
                print(msg, flush=True)

        elapsed = time.monotonic() - t_start
        if is_last:
            print(f"[rank {rank}] epoch {epoch}/{config.epochs} loss={losses[-1]:.4f} elapsed={elapsed:.1f}s", flush=True)
        else:
            print(f"[rank {rank}] epoch {epoch}/{config.epochs} elapsed={elapsed:.1f}s", flush=True)

    total_elapsed = time.monotonic() - t_start
    print(f"[rank {rank}] DPO training DONE in {total_elapsed:.1f}s ({total_steps} steps)", flush=True)
    if is_last and losses:
        print(f"[rank {rank}] loss avg first {min(8,len(losses))} steps: {sum(losses[:8])/min(8,len(losses)):.4f}", flush=True)
        print(f"[rank {rank}] loss avg last {min(8,len(losses))} steps:  {sum(losses[-8:])/min(8,len(losses)):.4f}", flush=True)

    _teardown(rank)
    return losses


# ---------------------------------------------------------------------------
# Shared: pipeline-parallel autoregressive generation
# ---------------------------------------------------------------------------


def _sample_token(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        return logits.argmax(dim=-1)
    probs = F.softmax(logits.float() / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def default_reward_fn(text: str) -> float:
    """Rule-based reward: rewards lexical diversity (small under-trained
    LMs sampled on-policy very commonly collapse into repetition loops -
    this directly penalizes that real, observable failure mode) plus a
    mild preference for landing near a target length. This is NOT a
    trained reward model - it's a real, deterministic scorer good enough
    to prove the GRPO/PPO MECHANISM (group-relative advantage / GAE
    actually moving the policy toward higher-scoring samples) at small
    scale. Swap in a real reward model by passing a different
    `reward_fn` to run_grpo_training/run_ppo_training directly."""
    tokens = text.split()
    if not tokens:
        return -1.0
    distinct_ratio = len(set(tokens)) / len(tokens)
    length_score = -abs(len(tokens) - 40) / 40.0
    return distinct_ratio + 0.3 * length_score


def pipeline_generate(rank, config, stage_layers, embed, norm, rotary, lm_head, hidden_size,
                       device, is_first, is_last, prev_rank, next_rank, tag_base,
                       group_size: int, max_new_tokens: int, temperature: float,
                       prompt_ids: torch.Tensor | None = None) -> torch.Tensor | None:
    """Token-by-token autoregressive decode across the pipeline, one full
    round trip per new token. Each stage keeps its own local KV cache
    (a transformers DynamicCache, indexed by each layer's own global
    layer_idx - correct without any special-casing since these ARE the
    original model's layer submodules) across steps, so only the
    embedding activation (rank_k -> rank_k+1) and the sampled token id
    (last stage -> rank 0, direct point-to-point since quic_dist is a
    full-mesh process group, not a chain) ever cross the network per
    step - never a KV cache itself.

    `prompt_ids` (only meaningful/required on rank 0): (1, prompt_len)
    int64 token ids for ONE prompt. It is replicated `group_size` times
    internally (not padded - identical rows need no attention mask), so
    sampling alone produces `group_size` distinct completions - this IS
    the "group" in GRPO/the rollout batch in PPO.

    Returns the full (group_size, max_new_tokens) generated-id tensor on
    rank 0 and on the last stage (the two ranks that actually need it -
    rank 0 to re-embed it in the follow-up teacher-forced training pass,
    the last stage as labels/reward input) - None on any middle rank."""
    from transformers.cache_utils import DynamicCache

    G = group_size
    world_size = config.world_size

    if is_first:
        prompt_len = prompt_ids.shape[1]
        len_t = torch.tensor([prompt_len], dtype=torch.long)
        for r in range(world_size):
            if r != rank:
                dist.send(len_t, dst=r, tag=tag_base)
    else:
        len_t = torch.zeros(1, dtype=torch.long)
        dist.recv(len_t, src=0, tag=tag_base)
        prompt_len = len_t.item()

    cache = DynamicCache()
    generated_chunks = []  # rank0 and is_last only; list of (G,) tensors, one per new token
    cur_token = None       # rank0 only: the (G,1) token to embed this call

    with torch.no_grad():
        for call_idx in range(max_new_tokens):
            is_prefill = call_idx == 0
            if is_prefill:
                seq_len_this = prompt_len
                cache_position = torch.arange(prompt_len, device=device)
            else:
                seq_len_this = 1
                cache_position = torch.tensor([prompt_len + call_idx - 1], device=device)
            position_ids = cache_position.unsqueeze(0).expand(G, -1)

            if is_first:
                tok_in = prompt_ids.expand(G, -1) if is_prefill else cur_token
                hidden = embed(tok_in.to(device)).to(config.torch_dtype)
            else:
                recv_buf = torch.zeros(G, seq_len_this, hidden_size, dtype=config.torch_dtype)
                dist.recv(recv_buf, src=prev_rank, tag=tag_base)
                hidden = recv_buf.to(device)

            pos_emb = rotary(hidden, position_ids) if rotary is not None else None
            out = hidden
            for layer in stage_layers:
                out = layer(out, attention_mask=None, position_ids=position_ids,
                            past_key_values=cache, use_cache=True, cache_position=cache_position,
                            position_embeddings=pos_emb)

            if is_last:
                last_hidden = out[:, -1:, :]
                logits = lm_head(norm(last_hidden).to(lm_head.weight.dtype))[:, -1, :]
                next_token = _sample_token(logits, temperature)  # (G,)
                generated_chunks.append(next_token.cpu())
                dist.send(next_token.cpu().unsqueeze(1), dst=0, tag=tag_base)
            else:
                dist.send(out.to(config.torch_dtype).cpu(), dst=next_rank, tag=tag_base)

            if is_first:
                recv_tok = torch.zeros(G, 1, dtype=torch.long)
                dist.recv(recv_tok, src=world_size - 1, tag=tag_base)
                cur_token = recv_tok
                generated_chunks.append(recv_tok.squeeze(1))

    if is_first or is_last:
        return torch.stack(generated_chunks, dim=1)  # (G, max_new_tokens)
    return None


# ---------------------------------------------------------------------------
# GRPO
# ---------------------------------------------------------------------------


@dataclass
class GRPOConfig(RLHFModelConfig):
    dataset_name: str = "tatsu-lab/alpaca"
    dataset_split: str = "train"
    num_examples: int | None = 32   # number of PROMPTS; each becomes one training step
    prompt_field: str = "instruction"
    max_prompt_len: int = 48

    group_size: int = 4       # completions sampled per prompt - "group" in GRPO
    max_new_tokens: int = 24
    temperature: float = 1.0
    kl_coef: float = 0.05
    epochs: int = 1

    @classmethod
    def from_file(cls, path: str) -> "GRPOConfig":
        return cls(**cls._load_raw(path))


def build_grpo_prompts(tokenizer, config: GRPOConfig) -> list[torch.Tensor]:
    from datasets import load_dataset

    split = config.dataset_split if config.num_examples is None else f"{config.dataset_split}[:{config.num_examples}]"
    ds = load_dataset(config.dataset_name, split=split)
    prompts = []
    for ex in ds:
        ids = tokenizer(ex[config.prompt_field], truncation=True, max_length=config.max_prompt_len,
                         add_special_tokens=True)["input_ids"]
        prompts.append(torch.tensor([ids], dtype=torch.long))
    return prompts


def run_grpo_training(rank: int, signaling_url: str, config: GRPOConfig, job_id: str = "grpo_pipeline",
                       reward_fn=default_reward_fn) -> list[float]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    local_gpu, device, peft_model, stage_layers, embed, norm, rotary, lm_head, hidden_size = _init_rank(
        rank, signaling_url, config, job_id
    )
    is_first = rank == 0
    is_last = rank == config.world_size - 1
    prev_rank = rank - 1 if rank > 0 else None
    next_rank = rank + 1 if rank < config.world_size - 1 else None

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr)

    prompts = build_grpo_prompts(tokenizer, config)
    n_steps = len(prompts)
    total_steps = n_steps * config.epochs
    G = config.group_size
    print(f"[rank {rank}] GRPO: {config.epochs} epochs x {n_steps} prompts = {total_steps} steps, "
          f"group_size={G}, max_new_tokens={config.max_new_tokens}", flush=True)

    losses: list[float] = []
    rewards_log: list[float] = []
    t_start = time.monotonic()
    step_counter = 0

    for epoch in range(config.epochs):
        for prompt_ids in prompts:
            step_counter += 1
            tag_gen = (step_counter % 4) * 3
            tag_ref = tag_gen + 1
            tag_policy = tag_gen + 2

            # --- 1. rollout: sample G completions from the CURRENT policy ---
            peft_model.eval()
            generated = pipeline_generate(
                rank, config, stage_layers, embed, norm, rotary, lm_head, hidden_size, device,
                is_first, is_last, prev_rank, next_rank, tag_gen,
                group_size=G, max_new_tokens=config.max_new_tokens, temperature=config.temperature,
                prompt_ids=prompt_ids if is_first else None,
            )
            peft_model.train()

            prompt_len = prompt_ids.shape[1]
            seq_len = prompt_len + config.max_new_tokens - 1  # predicting next-token, same -1 convention as elsewhere

            if is_first:
                full_ids = torch.cat([prompt_ids.expand(G, -1), generated], dim=1).to(device)  # (G, prompt_len+N)

            # --- 2. reward + group-relative advantage (last stage only; no
            # critic - the group mean/std IS the baseline, the defining
            # trick that lets GRPO drop PPO's value network) ---
            if is_last:
                texts = [tokenizer.decode(generated[g], skip_special_tokens=True) for g in range(G)]
                rewards = torch.tensor([reward_fn(t) for t in texts], dtype=torch.float32)
                advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
                rewards_log.append(rewards.mean().item())

            # --- 3. teacher-forced forward over prompt+generated: policy
            # (grad) and reference (no grad, adapters disabled - same
            # free-reference trick as DPO) - to get the on-policy logp of
            # exactly the tokens that were just sampled. ---
            def forced_forward(tag, no_grad):
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(G, -1)
                ctx = torch.no_grad() if no_grad else torch.enable_grad()
                with ctx:
                    if is_first:
                        hidden = embed(full_ids[:, :-1]).to(config.torch_dtype)
                        hidden_in = None
                    else:
                        recv_buf = torch.zeros(G, seq_len, hidden_size, dtype=config.torch_dtype)
                        dist.recv(recv_buf, src=prev_rank, tag=tag)
                        hidden_in = recv_buf.to(device)
                        if not no_grad:
                            hidden_in = hidden_in.requires_grad_(True)
                        hidden = hidden_in
                    pos_emb = rotary(hidden, position_ids) if rotary is not None else None
                    out = hidden
                    for layer in stage_layers:
                        out = layer(out, attention_mask=None, position_ids=position_ids, position_embeddings=pos_emb)
                    if is_last:
                        out = norm(out)
                        logits = lm_head(out.to(lm_head.weight.dtype))
                        return logits, hidden_in
                    dist.send(out.detach().to(config.torch_dtype).cpu(), dst=next_rank, tag=tag)
                    return out, hidden_in

            optimizer.zero_grad()
            with peft_model.disable_adapter():
                ref_out, _ = forced_forward(tag_ref, no_grad=True)
            policy_out, hidden_in = forced_forward(tag_policy, no_grad=False)

            if is_last:
                resp_start = prompt_len - 1  # logits[:, resp_start] predicts the first generated token
                resp_logits_p = policy_out[:, resp_start:, :]
                resp_logits_r = ref_out[:, resp_start:, :]
                resp_labels = generated.to(device)  # full_ids[:, prompt_len:] by construction - see pipeline_generate's docstring
                logp_policy = F.log_softmax(resp_logits_p.float(), dim=-1).gather(-1, resp_labels.unsqueeze(-1)).squeeze(-1)
                logp_ref = F.log_softmax(resp_logits_r.float(), dim=-1).gather(-1, resp_labels.unsqueeze(-1)).squeeze(-1)
                kl = (logp_policy - logp_ref)  # (G, N) - k1 estimator, per-token

                # Single-rollout-per-update simplification: the policy that
                # generated these samples IS the policy being updated (no
                # multiple inner PPO-clip epochs, unlike run_ppo_training
                # below), so the importance ratio is exactly 1 and this
                # reduces to REINFORCE with a group-relative baseline plus
                # a KL-to-reference penalty - the real GRPO objective minus
                # its clip term, which only does something across >1 update
                # per rollout.
                pg_loss = -(advantage.to(device).unsqueeze(1) * logp_policy).mean()
                loss = pg_loss + config.kl_coef * kl.mean()
                loss.backward()
                losses.append(loss.item())
            else:
                grad = torch.zeros(G, seq_len, hidden_size, dtype=config.torch_dtype)
                dist.recv(grad, src=next_rank, tag=tag_policy)
                policy_out.backward(grad.to(device))

            if not is_first:
                dist.send(hidden_in.grad.detach().to(config.torch_dtype).cpu(), dst=prev_rank, tag=tag_policy)

            optimizer.step()
            if step_counter <= 3 or step_counter % config.log_every == 0:
                msg = f"[rank {rank}] step {step_counter}/{total_steps}"
                if is_last:
                    msg += f" grpo_loss={losses[-1]:.4f} reward_mean={rewards_log[-1]:.3f} kl={kl.mean().item():.4f}"
                print(msg, flush=True)

        elapsed = time.monotonic() - t_start
        if is_last:
            print(f"[rank {rank}] epoch {epoch}/{config.epochs} loss={losses[-1]:.4f} "
                  f"reward={rewards_log[-1]:.3f} elapsed={elapsed:.1f}s", flush=True)
        else:
            print(f"[rank {rank}] epoch {epoch}/{config.epochs} elapsed={elapsed:.1f}s", flush=True)

    total_elapsed = time.monotonic() - t_start
    print(f"[rank {rank}] GRPO training DONE in {total_elapsed:.1f}s ({total_steps} steps)", flush=True)
    if is_last and rewards_log:
        print(f"[rank {rank}] reward avg first {min(8,len(rewards_log))} steps: {sum(rewards_log[:8])/min(8,len(rewards_log)):.4f}", flush=True)
        print(f"[rank {rank}] reward avg last {min(8,len(rewards_log))} steps:  {sum(rewards_log[-8:])/min(8,len(rewards_log)):.4f}", flush=True)

    _teardown(rank)
    return losses


# ---------------------------------------------------------------------------
# PPO
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig(RLHFModelConfig):
    dataset_name: str = "tatsu-lab/alpaca"
    dataset_split: str = "train"
    num_examples: int | None = 16
    prompt_field: str = "instruction"
    max_prompt_len: int = 32

    group_size: int = 4        # parallel rollouts per prompt (PPO doesn't need a group for the
                                # baseline the way GRPO does - the value head is the baseline -
                                # but reuses pipeline_generate's group-batching for real throughput)
    max_new_tokens: int = 16
    temperature: float = 1.0

    kl_coef: float = 0.05
    gamma: float = 1.0         # short single-turn completions - no real reason to discount within one
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ppo_epochs: int = 2        # inner clipped-update passes over ONE rollout batch - this is what
                                # actually exercises the clip (ratio==1 on the first inner pass, then
                                # diverges as the policy is updated) - GRPO above deliberately skips this
    epochs: int = 1
    vf_lr: float | None = None  # defaults to config.lr if unset

    @classmethod
    def from_file(cls, path: str) -> "PPOConfig":
        return cls(**cls._load_raw(path))


def build_ppo_prompts(tokenizer, config: PPOConfig) -> list[torch.Tensor]:
    from datasets import load_dataset

    split = config.dataset_split if config.num_examples is None else f"{config.dataset_split}[:{config.num_examples}]"
    ds = load_dataset(config.dataset_name, split=split)
    prompts = []
    for ex in ds:
        ids = tokenizer(ex[config.prompt_field], truncation=True, max_length=config.max_prompt_len,
                         add_special_tokens=True)["input_ids"]
        prompts.append(torch.tensor([ids], dtype=torch.long))
    return prompts


def _gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float):
    """rewards/values: (G, N). Standard token-level GAE, bootstrapped
    with a 0 terminal value (the episode - one generated completion -
    always ends at position N-1, there's no continuation to bootstrap
    from)."""
    G, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(G, device=rewards.device)
    for t in reversed(range(N)):
        next_value = values[:, t + 1] if t + 1 < N else torch.zeros(G, device=rewards.device)
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        gae = delta + gamma * lam * gae
        advantages[:, t] = gae
    returns = advantages + values
    return advantages, returns


def run_ppo_training(rank: int, signaling_url: str, config: PPOConfig, job_id: str = "ppo_pipeline",
                      reward_fn=default_reward_fn) -> list[float]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    local_gpu, device, peft_model, stage_layers, embed, norm, rotary, lm_head, hidden_size = _init_rank(
        rank, signaling_url, config, job_id
    )
    is_first = rank == 0
    is_last = rank == config.world_size - 1
    prev_rank = rank - 1 if rank > 0 else None
    next_rank = rank + 1 if rank < config.world_size - 1 else None

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    # Value head lives ONLY on the last stage, right next to lm_head - it
    # reads the same post-norm hidden states lm_head does. Kept in fp32
    # (a fresh, small, untrained head is the least numerically stable
    # part of this whole pipeline - not worth quantizing/downcasting).
    value_head = None
    if is_last:
        value_head = nn.Linear(hidden_size, 1).to(device=device, dtype=torch.float32)
        trainable = trainable + list(value_head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=config.lr)

    prompts = build_ppo_prompts(tokenizer, config)
    n_steps = len(prompts)
    total_steps = n_steps * config.epochs
    G = config.group_size
    print(f"[rank {rank}] PPO: {config.epochs} epochs x {n_steps} prompts = {total_steps} steps, "
          f"group_size={G}, max_new_tokens={config.max_new_tokens}, ppo_epochs={config.ppo_epochs}", flush=True)

    losses: list[float] = []
    rewards_log: list[float] = []
    t_start = time.monotonic()
    step_counter = 0

    for epoch in range(config.epochs):
        for prompt_ids in prompts:
            step_counter += 1
            tag_gen = (step_counter % 3) * (2 + config.ppo_epochs)
            tag_ref = tag_gen + 1
            tag_old = tag_gen + 2

            # --- 1. rollout: sample G completions from the CURRENT policy ---
            peft_model.eval()
            generated = pipeline_generate(
                rank, config, stage_layers, embed, norm, rotary, lm_head, hidden_size, device,
                is_first, is_last, prev_rank, next_rank, tag_gen,
                group_size=G, max_new_tokens=config.max_new_tokens, temperature=config.temperature,
                prompt_ids=prompt_ids if is_first else None,
            )
            peft_model.train()

            prompt_len = prompt_ids.shape[1]
            seq_len = prompt_len + config.max_new_tokens - 1
            resp_start = prompt_len - 1

            if is_first:
                full_ids = torch.cat([prompt_ids.expand(G, -1), generated], dim=1).to(device)

            def forced_forward(tag, no_grad, want_value):
                """Returns (logits_or_out, value_or_None, hidden_in) on
                the last stage; (out, None, hidden_in) elsewhere."""
                position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(G, -1)
                ctx = torch.no_grad() if no_grad else torch.enable_grad()
                with ctx:
                    if is_first:
                        hidden = embed(full_ids[:, :-1]).to(config.torch_dtype)
                        hidden_in = None
                    else:
                        recv_buf = torch.zeros(G, seq_len, hidden_size, dtype=config.torch_dtype)
                        dist.recv(recv_buf, src=prev_rank, tag=tag)
                        hidden_in = recv_buf.to(device)
                        if not no_grad:
                            hidden_in = hidden_in.requires_grad_(True)
                        hidden = hidden_in
                    pos_emb = rotary(hidden, position_ids) if rotary is not None else None
                    out = hidden
                    for layer in stage_layers:
                        out = layer(out, attention_mask=None, position_ids=position_ids, position_embeddings=pos_emb)
                    if is_last:
                        normed = norm(out)
                        logits = lm_head(normed.to(lm_head.weight.dtype))
                        value = value_head(normed[:, resp_start:, :].float()).squeeze(-1) if want_value else None
                        return logits, value, hidden_in
                    dist.send(out.detach().to(config.torch_dtype).cpu(), dst=next_rank, tag=tag)
                    return out, None, hidden_in

            # --- 2. reference logp (free via disable_adapter) and
            # OLD policy logp+value (snapshot before any update this
            # step) - both no-grad, both computed ONCE per rollout. ---
            with peft_model.disable_adapter():
                ref_out, _, _ = forced_forward(tag_ref, no_grad=True, want_value=False)
            old_out, old_value, _ = forced_forward(tag_old, no_grad=True, want_value=True)

            if is_last:
                resp_labels = generated.to(device)
                logp_ref = F.log_softmax(ref_out[:, resp_start:, :].float(), dim=-1).gather(-1, resp_labels.unsqueeze(-1)).squeeze(-1)
                logp_old = F.log_softmax(old_out[:, resp_start:, :].float(), dim=-1).gather(-1, resp_labels.unsqueeze(-1)).squeeze(-1)

                texts = [tokenizer.decode(generated[g], skip_special_tokens=True) for g in range(G)]
                env_reward = torch.tensor([reward_fn(t) for t in texts], dtype=torch.float32, device=device)
                rewards_log.append(env_reward.mean().item())

                # Per-token reward = KL-to-reference penalty everywhere,
                # plus the scalar task reward added only at the final
                # generated token - the standard "RLHF as a bandit with a
                # per-token KL tax" formulation.
                per_token_reward = -config.kl_coef * (logp_old - logp_ref)
                per_token_reward[:, -1] = per_token_reward[:, -1] + env_reward
                advantages, returns = _gae(per_token_reward.detach(), old_value.detach(), config.gamma, config.gae_lambda)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)

            # --- 3. ppo_epochs clipped-surrogate update passes over this
            # SAME rollout - every rank loops this many times in lockstep
            # (no communication needed to agree on the count, it's the
            # same config on every rank). ---
            last_loss = None
            for inner in range(config.ppo_epochs):
                tag_epoch = tag_old + 1 + inner
                optimizer.zero_grad()
                new_out, new_value, hidden_in = forced_forward(tag_epoch, no_grad=False, want_value=True)

                if is_last:
                    logp_new = F.log_softmax(new_out[:, resp_start:, :].float(), dim=-1).gather(-1, resp_labels.unsqueeze(-1)).squeeze(-1)
                    ratio = torch.exp(logp_new - logp_old.detach())
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(new_value, returns.detach())
                    loss = policy_loss + config.vf_coef * value_loss
                    loss.backward()
                    last_loss = loss.item()
                else:
                    grad = torch.zeros(G, seq_len, hidden_size, dtype=config.torch_dtype)
                    dist.recv(grad, src=next_rank, tag=tag_epoch)
                    new_out.backward(grad.to(device))

                if not is_first:
                    dist.send(hidden_in.grad.detach().to(config.torch_dtype).cpu(), dst=prev_rank, tag=tag_epoch)

                optimizer.step()

            if is_last:
                losses.append(last_loss)
            if step_counter <= 3 or step_counter % config.log_every == 0:
                msg = f"[rank {rank}] step {step_counter}/{total_steps}"
                if is_last:
                    msg += f" ppo_loss={losses[-1]:.4f} reward_mean={rewards_log[-1]:.3f}"
                print(msg, flush=True)

        elapsed = time.monotonic() - t_start
        if is_last:
            print(f"[rank {rank}] epoch {epoch}/{config.epochs} loss={losses[-1]:.4f} "
                  f"reward={rewards_log[-1]:.3f} elapsed={elapsed:.1f}s", flush=True)
        else:
            print(f"[rank {rank}] epoch {epoch}/{config.epochs} elapsed={elapsed:.1f}s", flush=True)

    total_elapsed = time.monotonic() - t_start
    print(f"[rank {rank}] PPO training DONE in {total_elapsed:.1f}s ({total_steps} steps)", flush=True)
    if is_last and rewards_log:
        print(f"[rank {rank}] reward avg first {min(8,len(rewards_log))} steps: {sum(rewards_log[:8])/min(8,len(rewards_log)):.4f}", flush=True)
        print(f"[rank {rank}] reward avg last {min(8,len(rewards_log))} steps:  {sum(rewards_log[-8:])/min(8,len(rewards_log)):.4f}", flush=True)

    _teardown(rank)
    return losses
