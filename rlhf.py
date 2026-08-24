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

from quic_dist.finetune import resolve_attr, stage_range, build_device_map, build_stage_model, run_decoder_layer


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


def _step_barrier(signaling_url, config, tag: str, job_id: str = "", timeout_s: int = 300):
    """A REAL per-step barrier - unlike `dist.barrier()`, which is only
    safe to call ONCE per process group's lifetime. Found via a real
    cross-machine GRPO run: PyTorch's own `_store_based_barrier` keys
    its store entry ONLY by the process group's name
    (`f"{STORE_BASED_BARRIER_PREFIX}:{group_name}"`, confirmed by
    reading its source directly), not by call site or call count. A
    SECOND `dist.barrier()` call in the same process group reads that
    already-satisfied key from the FIRST call and returns immediately -
    a real, silent no-op providing zero actual synchronization, which
    is exactly why adding a per-step `dist.barrier()` after the
    existing load-time one did not fix the timeout it was meant to fix.
    This reimplements the same add-then-wait-for-last-worker pattern
    directly against the store, with a fresh key per `tag` so each call
    is genuinely independent of every other barrier in the run.

    `job_id` is included in the key on top of `tag` - found via a real
    hang (distillation module, world_size=2): the signaling server's
    `/kv/*` store is shared across EVERY run that ever points at it, not
    scoped per process group. Two DIFFERENT runs that happen to reach
    the same `tag` (e.g. both hit "distill_1" - trivially likely across
    repeated dev-loop restarts against one long-lived local signaling
    server) silently share the same counter key. A crashed/killed EARLIER
    run's orphaned `add()` calls accumulate on that key forever (nothing
    ever consumes/resets it), pushing the count past `config.world_size`
    - and since the check is `count == world_size` (exact equality, not
    `>=`), a THIS-run's count that lands on an already-inflated key can
    permanently miss the equality and every rank hangs until its own
    300s timeout, even though both ranks genuinely reached the barrier.
    Confirmed directly: the signaling server's access log showed 20
    accumulated `/kv/add` calls against one `tag` from earlier crashed
    attempts before the hang. Scoping by `job_id` (already unique per
    run in every caller here) prevents different runs from ever sharing
    a key, independent of what `tag` values they happen to reach."""
    from quic_dist.store import QuicRendezvousStore

    store = QuicRendezvousStore(signaling_url, timeout=timedelta(seconds=timeout_s))
    key = f"step_barrier_{job_id}_{tag}"
    count = store.add(key, 1)
    if count == config.world_size:
        store.set(f"{key}:done", "1")
    store.wait([f"{key}:done"])


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

    # Real bug found via a cross-machine 27B run: model loading (esp.
    # quantized, multi-GB) takes very different real wall-clock time per
    # rank (disk speed, machine load) - without a barrier here, a
    # fast-loading rank can reach the training loop and start sending
    # real tensors while a slow-loading rank is still mid-load, and the
    # underlying QUIC connection's own idle timeout closes the
    # connection before the slow rank ever gets there. This barrier
    # forces every rank to finish loading before ANY of them proceeds.
    dist.barrier()
    print(f"[rank {rank}] all ranks finished loading, starting training", flush=True)

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
                out = run_decoder_layer(layer, out, attention_mask=None, position_ids=position_ids, position_embeddings=pos_emb)

            if is_last:
                out = norm(out)
                logits = lm_head(out.to(lm_head.weight.dtype))
                return logits, hidden_in
            else:
                dist.send(out.detach().to(config.torch_dtype).cpu(), dst=next_rank, tag=tag)
                return out, hidden_in

    for epoch in range(config.epochs):
        for b in range(n_steps):
            # Real bug found via a cross-machine 27B GRPO run (same
            # class as the load-time barrier, different location): one
            # rank finishing a step's bookkeeping faster than another
            # leaves the SLOWER rank's peer connection idle long enough
            # for the underlying QUIC connection's own idle timeout to
            # fire before the next step's first send/recv - DPO's single
            # forward+backward per step never left enough of a gap to
            # trigger this, but it's a real risk on any step whose
            # per-rank work is uneven or slow, so it's applied here too
            # even though this exact run didn't fail. Uses _step_barrier,
            # NOT dist.barrier() - see that function's docstring for why
            # a second dist.barrier() call is a silent no-op.
            _step_barrier(signaling_url, config, f"dpo_{epoch}_{b}", job_id=job_id)
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
# Reward Model (RM) - outcome reward model, Bradley-Terry pairwise loss,
# ONE scalar reward per full sequence. Trained checkpoints from this
# section are what makes run_ppo_training/run_grpo_training's `rm=`
# argument real: instead of default_reward_fn's rule-based text
# heuristic, PPO/GRPO can score rollouts with an actual trained model
# via a genuine pipeline-parallel forward pass (pipeline_score(),
# below the PRM section) - this is the "end-to-end RLHF" path.
# ---------------------------------------------------------------------------


@dataclass
class RMConfig(RLHFModelConfig):
    dataset_name: str = "Intel/orca_dpo_pairs"
    dataset_split: str = "train"
    num_examples: int | None = 64
    prompt_field: str = "question"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"

    max_prompt_len: int = 64
    max_response_len: int = 64

    # Small centering regularizer (InstructGPT-style): the Bradley-Terry
    # loss below only ever supervises the DIFFERENCE r_chosen-r_rejected,
    # so the reward scale itself is free to drift arbitrarily without
    # this - a real, well-known stabilization term, not defensive
    # programming for a case that can't happen.
    reward_l2_coef: float = 0.001

    batch: int = 1  # pairs per step; pipeline batch is 2x this (chosen+rejected concatenated)
    epochs: int = 1

    # Directory to save this rank's trained LoRA adapter + (last stage
    # only) reward_head to - see save_reward_stage(). None = don't save
    # (e.g. a pure mechanism-check run).
    save_path: str | None = None

    @classmethod
    def from_file(cls, path: str) -> "RMConfig":
        return cls(**cls._load_raw(path))

    @property
    def seq_len(self) -> int:
        return self.max_prompt_len + self.max_response_len


def _last_marked_idx(mask: torch.Tensor) -> torch.Tensor:
    """mask: (B, T) 0/1, the 1s forming a single contiguous run (e.g.
    build_dpo_dataset's response region). Returns (B,) the index of the
    LAST 1 in each row - the position whose hidden state has attended
    to the entire prompt+response, i.e. the position a reward head
    should read from. Multiplying by an increasing arange and taking
    argmax finds the last (highest-index) 1; plain mask.argmax(dim=1)
    would find the FIRST 1 instead, which is wrong here."""
    T = mask.shape[1]
    positions = torch.arange(T, device=mask.device).unsqueeze(0)
    return (mask * positions).argmax(dim=1)


def save_reward_stage(rank: int, config, peft_model, reward_head, save_dir: str) -> None:
    """Saves this rank's own slice of a trained reward model (an ORM
    from run_rm_training, or a PRM from run_prm_training - identical
    physical shape, only the training objective differs): the LoRA
    adapter via peft's own save_pretrained, plus, on the last stage
    only (where reward_head is not None), the reward head's raw
    weights. One subdirectory per rank since every rank only ever
    holds ITS OWN layer slice, mirroring build_stage_model - see
    load_reward_model() for the matching loader."""
    from pathlib import Path

    rank_dir = Path(save_dir) / f"rank{rank}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(rank_dir))
    if reward_head is not None:
        torch.save(reward_head.state_dict(), rank_dir / "reward_head.pt")
    print(f"[rank {rank}] saved reward model checkpoint to {rank_dir}", flush=True)


def run_rm_training(rank: int, signaling_url: str, config: RMConfig, job_id: str = "rm_pipeline") -> list[float]:
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

    dist.barrier()
    print(f"[rank {rank}] all ranks finished loading, starting training", flush=True)

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    # Reward head lives ONLY on the last stage, right next to
    # norm/lm_head - same placement pattern as PPO's value_head (both
    # read the same post-norm hidden states, both are a fresh
    # untrained fp32 head - not worth quantizing/downcasting).
    reward_head = None
    if is_last:
        reward_head = nn.Linear(hidden_size, 1).to(device=device, dtype=torch.float32)
        trainable = trainable + list(reward_head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=config.lr)

    # Reused UNCHANGED from the DPO section above - RMConfig declares
    # the exact same field names (prompt_field/chosen_field/
    # rejected_field/max_prompt_len/max_response_len/seq_len/batch/
    # dataset_name/dataset_split/num_examples) build_dpo_dataset needs,
    # and the (input_ids, response_mask) shape it returns is exactly
    # what RM training needs too - real reuse, not a copy.
    input_ids_all, response_mask_all = build_dpo_dataset(tokenizer, config)
    n_steps = input_ids_all.shape[0]
    total_steps = n_steps * config.epochs
    pbatch = 2 * config.batch  # chosen+rejected concatenated
    seq_len = config.seq_len   # no -1 shift here: RM pools ONE hidden state per
                                # sequence, it never predicts next-token logits
    print(f"[rank {rank}] RM: {config.epochs} epochs x {n_steps} steps/epoch = {total_steps} steps, "
          f"pipeline batch {pbatch} (={config.batch} pairs x2)", flush=True)

    losses: list[float] = []
    t_start = time.monotonic()
    step_counter = 0

    for epoch in range(config.epochs):
        for b in range(n_steps):
            _step_barrier(signaling_url, config, f"rm_{epoch}_{b}", job_id=job_id)
            step_counter += 1
            tag = step_counter % 8
            block = input_ids_all[b]
            mask = response_mask_all[b].to(device)
            optimizer.zero_grad()

            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(pbatch, -1)
            if is_first:
                hidden = embed(block.to(device)).to(config.torch_dtype)
                hidden_in = None
            else:
                recv_buf = torch.zeros(pbatch, seq_len, hidden_size, dtype=config.torch_dtype)
                dist.recv(recv_buf, src=prev_rank, tag=tag)
                hidden_in = recv_buf.to(device).requires_grad_(True)
                hidden = hidden_in

            pos_emb = rotary(hidden, position_ids) if rotary is not None else None
            out = hidden
            for layer in stage_layers:
                out = run_decoder_layer(layer, out, attention_mask=None, position_ids=position_ids, position_embeddings=pos_emb)

            if is_last:
                normed = norm(out)
                reward_all = reward_head(normed.float()).squeeze(-1)  # (pbatch, seq_len)
                idx = _last_marked_idx(mask)                          # (pbatch,)
                reward = reward_all.gather(1, idx.unsqueeze(1)).squeeze(1)  # (pbatch,)

                B = config.batch
                r_chosen, r_rejected = reward[:B], reward[B:]
                bt_loss = -F.logsigmoid(r_chosen - r_rejected).mean()
                l2 = (r_chosen.pow(2) + r_rejected.pow(2)).mean()
                loss = bt_loss + config.reward_l2_coef * l2
                loss.backward()
                losses.append(loss.item())
            else:
                dist.send(out.detach().to(config.torch_dtype).cpu(), dst=next_rank, tag=tag)
                grad = torch.zeros(pbatch, seq_len, hidden_size, dtype=config.torch_dtype)
                dist.recv(grad, src=next_rank, tag=tag)
                out.backward(grad.to(device))

            if not is_first:
                dist.send(hidden_in.grad.detach().to(config.torch_dtype).cpu(), dst=prev_rank, tag=tag)

            optimizer.step()
            if step_counter <= 3 or step_counter % config.log_every == 0:
                msg = f"[rank {rank}] step {step_counter}/{total_steps}"
                if is_last:
                    acc = (r_chosen > r_rejected).float().mean().item()
                    msg += f" rm_loss={losses[-1]:.4f} acc={acc:.3f}"
                print(msg, flush=True)

        elapsed = time.monotonic() - t_start
        if is_last:
            print(f"[rank {rank}] epoch {epoch}/{config.epochs} loss={losses[-1]:.4f} elapsed={elapsed:.1f}s", flush=True)
        else:
            print(f"[rank {rank}] epoch {epoch}/{config.epochs} elapsed={elapsed:.1f}s", flush=True)

    total_elapsed = time.monotonic() - t_start
    print(f"[rank {rank}] RM training DONE in {total_elapsed:.1f}s ({total_steps} steps)", flush=True)
    if is_last and losses:
        print(f"[rank {rank}] loss avg first {min(8,len(losses))} steps: {sum(losses[:8])/min(8,len(losses)):.4f}", flush=True)
        print(f"[rank {rank}] loss avg last {min(8,len(losses))} steps:  {sum(losses[-8:])/min(8,len(losses)):.4f}", flush=True)

    if config.save_path:
        save_reward_stage(rank, config, peft_model, reward_head, config.save_path)

    _teardown(rank)
    return losses


# ---------------------------------------------------------------------------
# Process Reward Model (PRM) - scores EACH reasoning step, not just the
# final outcome. Trained on peiyi9979/Math-Shepherd (real, verified
# schema: `label` is the same solution text as `input` but with each
# step's trailing "ки" marker replaced by "+"/"-" for correct/
# incorrect - confirmed by loading the dataset directly, not assumed
# from a stale description). One scalar prediction PER STEP-BOUNDARY
# token position, trained with per-step binary cross-entropy - the same
# approach PRM800K/Math-Shepherd's own papers use (a step's classifier
# token sits right after that step's content; this implementation reads
# the score at the LAST content token of each step instead of adding a
# separate classifier token, functionally equivalent and simpler since
# it needs no vocabulary/embedding changes).
# ---------------------------------------------------------------------------


@dataclass
class PRMConfig(RLHFModelConfig):
    dataset_name: str = "peiyi9979/Math-Shepherd"
    dataset_split: str = "train"
    num_examples: int | None = 64
    label_field: str = "label"       # already has per-step "+"/"-" applied - see module docstring above
    step_positive_marker: str = "+"
    step_negative_marker: str = "-"

    max_len: int = 256     # full prompt+steps token budget
    max_steps: int = 8     # cap step-boundary positions scored per example; extra steps truncated

    epochs: int = 1
    save_path: str | None = None

    @classmethod
    def from_file(cls, path: str) -> "PRMConfig":
        return cls(**cls._load_raw(path))

    @property
    def seq_len(self) -> int:
        return self.max_len


def _parse_math_shepherd_label(label_text: str, pos_marker: str, neg_marker: str):
    """Math-Shepherd's `label` field is the solution broken into lines,
    one step per line, each ending in " +" or " -" (see module
    docstring). Returns (step_texts, step_labels) - step_texts has the
    trailing marker stripped, step_labels is 1 for pos_marker, 0 for
    neg_marker. A line ending in neither is skipped (malformed/
    boundary artifact, not assumed to happen but not fatal if it does)."""
    step_texts, step_labels = [], []
    for line in label_text.split("\n"):
        line = line.rstrip()
        if line.endswith(pos_marker):
            step_labels.append(1)
            step_texts.append(line[: -len(pos_marker)].rstrip())
        elif line.endswith(neg_marker):
            step_labels.append(0)
            step_texts.append(line[: -len(neg_marker)].rstrip())
    return step_texts, step_labels


def build_prm_dataset(tokenizer, config: PRMConfig) -> list[dict]:
    """Returns a flat list of {input_ids: (T,), step_positions: (S,),
    step_labels: (S,)} - kept ragged (not stacked into one fixed-shape
    tensor like build_dataset/build_dpo_dataset) because both T and S
    genuinely vary per example. run_prm_training processes one example
    per pipeline step (batch=1) specifically so this ragged shape never
    needs padding/collation - see its docstring."""
    from datasets import load_dataset

    split = config.dataset_split if config.num_examples is None else f"{config.dataset_split}[:{config.num_examples}]"
    ds = load_dataset(config.dataset_name, split=split)

    examples = []
    for ex in ds:
        step_texts, step_labels = _parse_math_shepherd_label(
            ex[config.label_field], config.step_positive_marker, config.step_negative_marker
        )
        if not step_texts:
            continue
        step_texts = step_texts[: config.max_steps]
        step_labels = step_labels[: config.max_steps]

        ids: list[int] = []
        step_positions: list[int] = []
        for text in step_texts:
            piece_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if not piece_ids:
                continue
            ids.extend(piece_ids)
            step_positions.append(len(ids) - 1)  # this step's LAST token index

        if not step_positions:
            continue
        if len(ids) > config.max_len:
            # Truncate to the token budget - drop any step whose
            # boundary position would fall outside it rather than
            # silently scoring at a clipped/wrong position.
            keep_n = sum(1 for pos in step_positions if pos < config.max_len)
            if keep_n == 0:
                continue
            step_positions = step_positions[:keep_n]
            step_labels = step_labels[:keep_n]
            ids = ids[: config.max_len]

        examples.append({
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "step_positions": torch.tensor(step_positions, dtype=torch.long),
            "step_labels": torch.tensor(step_labels, dtype=torch.float32),
        })
    return examples


def run_prm_training(rank: int, signaling_url: str, config: PRMConfig, job_id: str = "prm_pipeline") -> list[float]:
    """Real batch=1-per-step training (see build_prm_dataset's
    docstring for why): every rank independently builds the SAME
    dataset deterministically (identical tokenizer, identical dataset,
    no randomness) - the exact same redundant-construction pattern
    run_dpo_training already relies on for input_ids_all, so shapes
    agree across ranks with zero network broadcast needed for this
    part (unlike pipeline_generate's prompt_len broadcast, which exists
    because ITS prompt genuinely originates on one specific rank)."""
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

    dist.barrier()
    print(f"[rank {rank}] all ranks finished loading, starting training", flush=True)

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    reward_head = None
    if is_last:
        # Same head shape as the ORM's reward_head above - PRM just
        # reads it at MULTIPLE positions per sequence (one per
        # reasoning step) instead of pooling once at the end.
        reward_head = nn.Linear(hidden_size, 1).to(device=device, dtype=torch.float32)
        trainable = trainable + list(reward_head.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=config.lr)

    examples = build_prm_dataset(tokenizer, config)
    n_steps = len(examples)
    total_steps = n_steps * config.epochs
    print(f"[rank {rank}] PRM: {config.epochs} epochs x {n_steps} examples = {total_steps} steps", flush=True)

    losses: list[float] = []
    t_start = time.monotonic()
    step_counter = 0

    for epoch in range(config.epochs):
        for ex in examples:
            _step_barrier(signaling_url, config, f"prm_{step_counter}", job_id=job_id)
            step_counter += 1
            tag = step_counter % 8
            input_ids = ex["input_ids"].unsqueeze(0)  # (1, T)
            step_positions = ex["step_positions"].to(device)
            step_labels = ex["step_labels"].to(device)
            T = input_ids.shape[1]
            optimizer.zero_grad()

            position_ids = torch.arange(T, device=device).unsqueeze(0)
            if is_first:
                hidden = embed(input_ids.to(device)).to(config.torch_dtype)
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

            if is_last:
                normed = norm(out)
                step_scores = reward_head(normed[0, step_positions, :].float()).squeeze(-1)  # (S,)
                loss = F.binary_cross_entropy_with_logits(step_scores, step_labels)
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
                    pred = (torch.sigmoid(step_scores) > 0.5).float()
                    acc = (pred == step_labels).float().mean().item()
                    msg += f" prm_loss={losses[-1]:.4f} step_acc={acc:.3f} n_steps={step_labels.numel()}"
                print(msg, flush=True)

        elapsed = time.monotonic() - t_start
        if is_last:
            print(f"[rank {rank}] epoch {epoch}/{config.epochs} loss={losses[-1]:.4f} elapsed={elapsed:.1f}s", flush=True)
        else:
            print(f"[rank {rank}] epoch {epoch}/{config.epochs} elapsed={elapsed:.1f}s", flush=True)

    total_elapsed = time.monotonic() - t_start
    print(f"[rank {rank}] PRM training DONE in {total_elapsed:.1f}s ({total_steps} steps)", flush=True)
    if is_last and losses:
        print(f"[rank {rank}] loss avg first {min(8,len(losses))} steps: {sum(losses[:8])/min(8,len(losses)):.4f}", flush=True)
        print(f"[rank {rank}] loss avg last {min(8,len(losses))} steps:  {sum(losses[-8:])/min(8,len(losses)):.4f}", flush=True)

    if config.save_path:
        save_reward_stage(rank, config, peft_model, reward_head, config.save_path)

    _teardown(rank)
    return losses


def load_reward_model(rank: int, local_gpu: int, rm_config, checkpoint_dir: str) -> "LoadedRewardModel":
    """Rebuilds this rank's slice of a previously-trained reward model
    (an ORM from run_rm_training, or a PRM from run_prm_training used
    ORM-style via pipeline_score - see LoadedRewardModel) for use as a
    REAL scorer during PPO/GRPO. This is what makes "end-to-end RLHF"
    real rather than a rule-based stand-in: the reward signal comes
    from an actual trained model's forward pass over the network.

    `rm_config` must be the SAME RMConfig/PRMConfig the checkpoint was
    trained with (model_path/num_layers/world_size/LoRA target_modules
    etc. all need to match - the same requirement any peft adapter
    reload has) AND its world_size/rank layout must match the CALLING
    training run's policy model 1:1 (rank k of both models must live on
    the same machine) - pipeline_score() below sends/receives against
    the SAME rank numbering the policy model's own pipeline uses.
    Kept frozen: every parameter has requires_grad_(False), model in
    eval() - this is a scorer, never trained further here."""
    from pathlib import Path
    from peft import PeftModel

    rank_dir = Path(checkpoint_dir) / f"rank{rank}"
    peft_model, _, _, _, _, _, hidden_size = build_stage_model(rank, local_gpu, rm_config)
    # build_stage_model always attaches a FRESH (randomly initialized)
    # LoRA adapter - discard it and load the ACTUAL trained one from
    # disk instead. The quantized base weights underneath are identical
    # either way (same model_path/quantization config), so there's no
    # need to reload those, only swap the small adapter.
    base_model = peft_model.base_model.model
    peft_model = PeftModel.from_pretrained(base_model, str(rank_dir), is_trainable=False)
    for p in peft_model.parameters():
        p.requires_grad_(False)
    peft_model.eval()

    layers = resolve_attr(peft_model.base_model.model, rm_config.layers_attr)
    my_range = stage_range(rank, rm_config)
    stage_layers = layers[my_range.start: my_range.stop]
    embed = resolve_attr(peft_model.base_model.model, rm_config.embed_attr)
    norm = resolve_attr(peft_model.base_model.model, rm_config.norm_attr)
    rotary = resolve_attr(peft_model.base_model.model, rm_config.rotary_attr) if rm_config.rotary_attr else None

    reward_head = None
    head_path = rank_dir / "reward_head.pt"
    if head_path.exists():
        reward_head = nn.Linear(hidden_size, 1).to(device=f"cuda:{local_gpu}", dtype=torch.float32)
        reward_head.load_state_dict(torch.load(head_path, map_location=f"cuda:{local_gpu}"))
        reward_head.eval()
        for p in reward_head.parameters():
            p.requires_grad_(False)

    return LoadedRewardModel(rm_config, stage_layers, embed, norm, rotary, hidden_size, reward_head)


@dataclass
class LoadedRewardModel:
    """A previously-trained reward model, reloaded (via
    load_reward_model()) and ready to score during PPO/GRPO through
    pipeline_score() - see that function and load_reward_model()'s
    docstring for the world_size/rank-layout requirement."""
    config: object
    stage_layers: object
    embed: object
    norm: object
    rotary: object
    hidden_size: int
    reward_head: object


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
    from transformers import AutoConfig
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

    # DynamicCache() with NO config builds an empty `self.layers` list
    # that only grows lazily on a plain KV `.update()` call - fine for a
    # uniform full-attention model (works on Qwen2.5-0.5B), but a real
    # crash (`IndexError: list index out of range` inside
    # `update_conv_state`) on a HYBRID linear/full-attention model
    # (Qwen3.8-27B): its linear_attention layers directly index
    # `self.layers[layer_idx]` expecting a pre-built
    # `LinearAttentionCacheLayerMixin` entry at that GLOBAL index, with
    # no lazy-grow fallback. Passing `config=` here makes DynamicCache
    # inspect the model's real per-layer types up front and build the
    # right cache-layer class for every one of them (found via direct
    # testing on a real cross-machine GRPO run, not anticipated).
    #
    # NOT every installed transformers accepts `config=` here - it was
    # added in a later release than this environment's own default (a
    # real TypeError hit directly: transformers==4.53.3 has no `config`
    # kwarg on DynamicCache.__init__ at all). Try the config-aware
    # constructor first (needed for hybrid-attention models); fall back
    # to the plain one on TypeError - correct for a uniform-attention
    # model on either transformers version, and the best available on
    # an old transformers even for a hybrid model (better than crashing
    # outright on an import that predates the fix this comment describes).
    hf_config = AutoConfig.from_pretrained(config.model_path)
    try:
        cache = DynamicCache(config=hf_config)
    except TypeError:
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
                # The cache kwarg's NAME has genuinely changed between
                # transformers versions (confirmed directly: this
                # environment's Qwen2DecoderLayer.forward takes
                # `past_key_value`, singular - an OLDER/newer version may
                # take `past_key_values`, plural). Passing the wrong name
                # does NOT raise - it's silently absorbed into the
                # layer's **kwargs (Unpack[FlashAttentionKwargs]) and the
                # cache is never actually threaded through, so decoding
                # would silently lose all cross-step context instead of
                # erroring - a real, dangerous-because-silent failure
                # mode, not a hypothetical one. Try both real kwarg
                # names rather than guessing one.
                try:
                    out = run_decoder_layer(layer, out, attention_mask=None, position_ids=position_ids,
                                             past_key_value=cache, use_cache=True, cache_position=cache_position,
                                             position_embeddings=pos_emb)
                except TypeError:
                    out = run_decoder_layer(layer, out, attention_mask=None, position_ids=position_ids,
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


def pipeline_score(rank, config, stage_layers, embed, norm, rotary, hidden_size, reward_head,
                    device, is_first, is_last, prev_rank, next_rank, tag,
                    input_ids: torch.Tensor | None = None) -> torch.Tensor | None:
    """One real pipeline-parallel forward pass through a FROZEN
    (no-grad) reward model (see LoadedRewardModel/load_reward_model),
    pooled at the LAST token position - the real end-to-end reward
    signal run_ppo_training/run_grpo_training's `rm=` argument uses
    instead of reward_fn's rule-based text scorer.

    `input_ids` (only meaningful/required on rank 0, same convention as
    pipeline_generate's `prompt_ids`): (B, T) int64. Pools at position
    T-1 rather than a response_mask because every caller here passes a
    FIXED-length, unpadded prompt+completion (pipeline_generate never
    early-stops or pads mid-batch) - T-1 IS the last real token by
    construction, no mask needed. Returns (B,) reward on the last
    stage, None elsewhere."""
    if is_first:
        B, T = input_ids.shape
        len_t = torch.tensor([B, T], dtype=torch.long)
        for r in range(config.world_size):
            if r != rank:
                dist.send(len_t, dst=r, tag=tag)
    else:
        len_t = torch.zeros(2, dtype=torch.long)
        dist.recv(len_t, src=0, tag=tag)
        B, T = len_t.tolist()

    position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    with torch.no_grad():
        if is_first:
            hidden = embed(input_ids.to(device)).to(config.torch_dtype)
        else:
            recv_buf = torch.zeros(B, T, hidden_size, dtype=config.torch_dtype)
            dist.recv(recv_buf, src=prev_rank, tag=tag)
            hidden = recv_buf.to(device)

        pos_emb = rotary(hidden, position_ids) if rotary is not None else None
        out = hidden
        for layer in stage_layers:
            out = run_decoder_layer(layer, out, attention_mask=None, position_ids=position_ids, position_embeddings=pos_emb)

        if is_last:
            normed = norm(out)
            reward_all = reward_head(normed.float()).squeeze(-1)  # (B, T)
            return reward_all[:, -1].cpu()
        else:
            dist.send(out.to(config.torch_dtype).cpu(), dst=next_rank, tag=tag)
            return None


def pipeline_score_prm(rank, config, stage_layers, embed, norm, rotary, hidden_size, reward_head,
                        device, is_first, is_last, prev_rank, next_rank, tag,
                        input_ids: torch.Tensor | None = None,
                        step_positions: torch.Tensor | None = None) -> torch.Tensor | None:
    """Same real pipeline-parallel frozen forward pass as
    pipeline_score(), but reads MULTIPLE per-step positions instead of
    pooling once at the last token - the natural way to use a trained
    PRM (run_prm_training) for real scoring: e.g. best-of-N reranking
    of N candidate solutions by their weakest (minimum) step score,
    the same use PRM800K/Math-Shepherd's own papers validate PRMs for.

    `input_ids` (B, T) and `step_positions` (B, S) are only
    meaningful/required on rank 0 - S must be the same across the
    batch (candidates with different real step counts should pad
    step_positions with a repeated index and have the caller mask that
    column out of whatever aggregation it does afterward; kept
    unpadded/simple here since a real caller usually scores one
    prompt's N candidates - a shared step-count budget - at a time).
    Returns (B, S) per-step reward on the last stage, None elsewhere."""
    if is_first:
        B, T = input_ids.shape
        S = step_positions.shape[1]
        len_t = torch.tensor([B, T, S], dtype=torch.long)
        for r in range(config.world_size):
            if r != rank:
                dist.send(len_t, dst=r, tag=tag)
                dist.send(step_positions, dst=r, tag=tag)
    else:
        len_t = torch.zeros(3, dtype=torch.long)
        dist.recv(len_t, src=0, tag=tag)
        B, T, S = len_t.tolist()
        step_positions = torch.zeros(B, S, dtype=torch.long)
        dist.recv(step_positions, src=0, tag=tag)

    position_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
    with torch.no_grad():
        if is_first:
            hidden = embed(input_ids.to(device)).to(config.torch_dtype)
        else:
            recv_buf = torch.zeros(B, T, hidden_size, dtype=config.torch_dtype)
            dist.recv(recv_buf, src=prev_rank, tag=tag)
            hidden = recv_buf.to(device)

        pos_emb = rotary(hidden, position_ids) if rotary is not None else None
        out = hidden
        for layer in stage_layers:
            out = run_decoder_layer(layer, out, attention_mask=None, position_ids=position_ids, position_embeddings=pos_emb)

        if is_last:
            normed = norm(out)
            reward_all = reward_head(normed.float()).squeeze(-1)  # (B, T)
            idx = step_positions.to(device)
            return reward_all.gather(1, idx).cpu()
        else:
            dist.send(out.to(config.torch_dtype).cpu(), dst=next_rank, tag=tag)
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
                       reward_fn=default_reward_fn, rm: "LoadedRewardModel | None" = None) -> list[float]:
    """`rm`, when given (see load_reward_model()), scores each rollout
    with a REAL trained reward model via pipeline_score() - a genuine
    pipeline-parallel forward pass every rank participates in - instead
    of reward_fn's rule-based text heuristic. This is the real
    end-to-end RLHF path: SFT -> run_rm_training -> load_reward_model
    -> this. `reward_fn` is ignored when `rm` is set."""
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

    # See run_dpo_training's identical barrier for why: without this, a
    # fast-loading rank can start sending real tensors before a
    # slow-loading rank finishes, and the QUIC connection's own idle
    # timeout closes it before the slow rank ever gets there.
    dist.barrier()
    print(f"[rank {rank}] all ranks finished loading, starting training", flush=True)

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
            # See run_dpo_training's identical per-step barrier for why:
            # a real cross-machine 27B run hit exactly this - one rank
            # reaching this step's generation recv before another rank
            # finished the previous step's bookkeeping, idle-timing-out
            # the underlying QUIC connection. Uses _step_barrier, NOT
            # dist.barrier() - see that function's docstring for why a
            # second dist.barrier() call is a silent no-op (this was the
            # actual root cause of the real timeout observed here, not
            # just a plausible-sounding guess - a first attempt using
            # plain dist.barrier() here did NOT fix the crash).
            _step_barrier(signaling_url, config, f"grpo_{step_counter + 1}", job_id=job_id)
            step_counter += 1
            tag_gen = (step_counter % 4) * 3
            tag_ref = tag_gen + 1
            tag_policy = tag_gen + 2
            # Deliberately far outside the tag_gen/tag_ref/tag_policy
            # range above rather than folded into that formula's own
            # modulo width - this is an EXISTING, already cross-machine-
            # validated tag scheme (see project memory); a fixed,
            # clearly-separated offset adds the one new tag pipeline_score
            # needs without touching arithmetic that already works.
            tag_rm = 1000 + (step_counter % 8)

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
            # trick that lets GRPO drop PPO's value network). Two reward
            # sources: `rm` (a REAL trained reward model, scored via a
            # genuine pipeline-parallel round trip EVERY rank must
            # participate in - see pipeline_score()) when given, else the
            # default rule-based reward_fn (text-only, is_last alone). ---
            if rm is not None:
                rm_input = full_ids.cpu() if is_first else None
                rm_reward = pipeline_score(
                    rank, rm.config, rm.stage_layers, rm.embed, rm.norm, rm.rotary, rm.hidden_size, rm.reward_head,
                    device, is_first, is_last, prev_rank, next_rank, tag_rm, input_ids=rm_input,
                )
                if is_last:
                    rewards = rm_reward
            elif is_last:
                texts = [tokenizer.decode(generated[g], skip_special_tokens=True) for g in range(G)]
                rewards = torch.tensor([reward_fn(t) for t in texts], dtype=torch.float32)

            if is_last:
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
                        out = run_decoder_layer(layer, out, attention_mask=None, position_ids=position_ids, position_embeddings=pos_emb)
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
# RLOO / REINFORCE
# ---------------------------------------------------------------------------


@dataclass
class RLOOConfig(RLHFModelConfig):
    """Structurally near-identical to GRPOConfig - same rollout
    (pipeline_generate), same free reference-model KL trick, same
    single-pass (no PPO clip) policy-gradient update. The ONE real
    difference is the advantage estimator, controlled by `baseline`:

    - "rloo" (the actual RLOO estimator, Kool et al. / used in the RLOO-
      for-RLHF line of work): advantage_i = r_i - mean(r_{-i}), i.e.
      each sample's baseline is the mean of every OTHER sample in its
      group, never itself. Unbiased (a real, provable property GRPO's
      group-mean-AND-std normalization does not have - GRPO trades that
      for variance reduction via the std division instead). Needs
      group_size >= 2. This is the default because it is the estimator
      actually named "RLOO" in the literature - "mean"/"none" below are
      REINFORCE variants offered from the same code path since the user
      asked for both.
    - "mean": plain REINFORCE with a group-mean baseline (includes the
      sample itself in the mean it's compared against - biased but the
      common practical simplification, no std division unlike GRPO).
    - "none": raw REINFORCE, advantage_i = r_i directly - the classic
      algorithm, kept for completeness even though its variance is
      usually too high to be practically useful past a handful of
      training steps.

    No std normalization in ANY of these three, unlike GRPOConfig's
    (rewards - mean) / std - that division is a GRPO-specific choice,
    not part of the original RLOO/REINFORCE formulations, so it is
    deliberately NOT reproduced here."""

    dataset_name: str = "tatsu-lab/alpaca"
    dataset_split: str = "train"
    num_examples: int | None = 32
    prompt_field: str = "instruction"
    max_prompt_len: int = 48

    group_size: int = 4
    max_new_tokens: int = 24
    temperature: float = 1.0
    kl_coef: float = 0.05
    baseline: str = "rloo"   # "rloo" | "mean" | "none"
    epochs: int = 1

    @classmethod
    def from_file(cls, path: str) -> "RLOOConfig":
        return cls(**cls._load_raw(path))


def build_rloo_prompts(tokenizer, config: RLOOConfig) -> list[torch.Tensor]:
    from datasets import load_dataset

    split = config.dataset_split if config.num_examples is None else f"{config.dataset_split}[:{config.num_examples}]"
    ds = load_dataset(config.dataset_name, split=split)
    prompts = []
    for ex in ds:
        ids = tokenizer(ex[config.prompt_field], truncation=True, max_length=config.max_prompt_len,
                         add_special_tokens=True)["input_ids"]
        prompts.append(torch.tensor([ids], dtype=torch.long))
    return prompts


def _rloo_advantage(rewards: torch.Tensor, baseline: str) -> torch.Tensor:
    """rewards: (G,). No std normalization in any branch - see
    RLOOConfig's docstring for why that's deliberate, not an oversight."""
    G = rewards.shape[0]
    if baseline == "rloo":
        if G < 2:
            raise ValueError(f"baseline='rloo' needs group_size >= 2 to leave one out, got {G}")
        total = rewards.sum()
        return rewards - (total - rewards) / (G - 1)
    if baseline == "mean":
        return rewards - rewards.mean()
    if baseline == "none":
        return rewards
    raise ValueError(f"unknown baseline {baseline!r} - expected 'rloo', 'mean', or 'none'")


def run_rloo_training(rank: int, signaling_url: str, config: RLOOConfig, job_id: str = "rloo_pipeline",
                       reward_fn=default_reward_fn, rm: "LoadedRewardModel | None" = None) -> list[float]:
    """`rm`, when given, scores each rollout with a real trained reward
    model via pipeline_score() instead of reward_fn's rule-based text
    heuristic - identical contract to run_grpo_training's `rm` param."""
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

    # See run_dpo_training's identical barrier for why.
    dist.barrier()
    print(f"[rank {rank}] all ranks finished loading, starting training", flush=True)

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.lr)

    prompts = build_rloo_prompts(tokenizer, config)
    n_steps = len(prompts)
    total_steps = n_steps * config.epochs
    G = config.group_size
    print(f"[rank {rank}] RLOO({config.baseline}): {config.epochs} epochs x {n_steps} prompts = {total_steps} steps, "
          f"group_size={G}, max_new_tokens={config.max_new_tokens}", flush=True)

    losses: list[float] = []
    rewards_log: list[float] = []
    t_start = time.monotonic()
    step_counter = 0

    for epoch in range(config.epochs):
        for prompt_ids in prompts:
            # See run_grpo_training's identical per-step barrier for why
            # this is _step_barrier and NOT plain dist.barrier().
            _step_barrier(signaling_url, config, f"rloo_{step_counter + 1}", job_id=job_id)
            step_counter += 1
            tag_gen = (step_counter % 4) * 3
            tag_ref = tag_gen + 1
            tag_policy = tag_gen + 2
            tag_rm = 1000 + (step_counter % 8)

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

            if is_first:
                full_ids = torch.cat([prompt_ids.expand(G, -1), generated], dim=1).to(device)

            # --- 2. reward + leave-one-out (or plain-mean/none) advantage.
            # Same two reward sources as GRPO: a real trained reward
            # model (`rm`) via a genuine pipeline-parallel round trip, or
            # the default rule-based reward_fn (text-only, is_last alone). ---
            if rm is not None:
                rm_input = full_ids.cpu() if is_first else None
                rm_reward = pipeline_score(
                    rank, rm.config, rm.stage_layers, rm.embed, rm.norm, rm.rotary, rm.hidden_size, rm.reward_head,
                    device, is_first, is_last, prev_rank, next_rank, tag_rm, input_ids=rm_input,
                )
                if is_last:
                    rewards = rm_reward
            elif is_last:
                texts = [tokenizer.decode(generated[g], skip_special_tokens=True) for g in range(G)]
                rewards = torch.tensor([reward_fn(t) for t in texts], dtype=torch.float32)

            if is_last:
                advantage = _rloo_advantage(rewards, config.baseline)
                rewards_log.append(rewards.mean().item())

            # --- 3. teacher-forced forward over prompt+generated: policy
            # (grad) and reference (no grad, adapters disabled) - same
            # free-reference trick as DPO/GRPO. ---
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
                        out = run_decoder_layer(layer, out, attention_mask=None, position_ids=position_ids, position_embeddings=pos_emb)
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
                resp_start = prompt_len - 1
                resp_logits_p = policy_out[:, resp_start:, :]
                resp_logits_r = ref_out[:, resp_start:, :]
                resp_labels = generated.to(device)
                logp_policy = F.log_softmax(resp_logits_p.float(), dim=-1).gather(-1, resp_labels.unsqueeze(-1)).squeeze(-1)
                logp_ref = F.log_softmax(resp_logits_r.float(), dim=-1).gather(-1, resp_labels.unsqueeze(-1)).squeeze(-1)
                kl = (logp_policy - logp_ref)  # (G, N) - k1 estimator, per-token

                # Plain REINFORCE/RLOO objective: -(advantage * logp).mean()
                # plus the same KL-to-reference regularizer GRPO uses -
                # ratio is exactly 1 here too (single update per rollout,
                # no PPO clip - see run_ppo_training for the mode that
                # actually needs the clip).
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
                    msg += f" rloo_loss={losses[-1]:.4f} reward_mean={rewards_log[-1]:.3f} kl={kl.mean().item():.4f}"
                print(msg, flush=True)

        elapsed = time.monotonic() - t_start
        if is_last:
            print(f"[rank {rank}] epoch {epoch}/{config.epochs} loss={losses[-1]:.4f} "
                  f"reward={rewards_log[-1]:.3f} elapsed={elapsed:.1f}s", flush=True)
        else:
            print(f"[rank {rank}] epoch {epoch}/{config.epochs} elapsed={elapsed:.1f}s", flush=True)

    total_elapsed = time.monotonic() - t_start
    print(f"[rank {rank}] RLOO training DONE in {total_elapsed:.1f}s ({total_steps} steps)", flush=True)
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
                      reward_fn=default_reward_fn, rm: "LoadedRewardModel | None" = None) -> list[float]:
    """`rm`, when given (see load_reward_model()), scores each rollout
    with a REAL trained reward model via pipeline_score() instead of
    reward_fn's rule-based text heuristic - see run_grpo_training's
    identical `rm` parameter docstring. `reward_fn` is ignored when
    `rm` is set."""
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

    # See run_dpo_training's identical barrier for why.
    dist.barrier()
    print(f"[rank {rank}] all ranks finished loading, starting training", flush=True)

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
            # See run_dpo_training's/run_grpo_training's identical
            # per-step barrier for why this is _step_barrier and NOT
            # plain dist.barrier() (which is a silent no-op past the
            # first call in a process group's lifetime).
            _step_barrier(signaling_url, config, f"ppo_{step_counter + 1}", job_id=job_id)
            step_counter += 1
            tag_gen = (step_counter % 3) * (2 + config.ppo_epochs)
            tag_ref = tag_gen + 1
            tag_old = tag_gen + 2
            # Same deliberately-far-outside-the-existing-scheme choice
            # as run_grpo_training's tag_rm - see that function's
            # comment for why this isn't folded into the formula above.
            tag_rm = 1000 + (step_counter % 8)

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

            # If `rm` is given, score the rollout with a REAL trained
            # reward model NOW via a genuine pipeline-parallel round
            # trip - every rank must participate (see pipeline_score()),
            # so this happens unconditionally, not inside `if is_last`.
            # The rule-based reward_fn path (below, is_last only) is
            # unchanged when rm is None.
            if rm is not None:
                rm_input = full_ids.cpu() if is_first else None
                rm_reward_out = pipeline_score(
                    rank, rm.config, rm.stage_layers, rm.embed, rm.norm, rm.rotary, rm.hidden_size, rm.reward_head,
                    device, is_first, is_last, prev_rank, next_rank, tag_rm, input_ids=rm_input,
                )

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
                        out = run_decoder_layer(layer, out, attention_mask=None, position_ids=position_ids, position_embeddings=pos_emb)
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

                if rm is not None:
                    env_reward = rm_reward_out.to(device)
                else:
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
