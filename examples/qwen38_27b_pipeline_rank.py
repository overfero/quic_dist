"""Real 4-stage pipeline-parallel QLoRA fine-tuning of a real ~27B
parameter model (Qwen/Qwen3.8-27B, architecture qwen3_5 - a hybrid
linear-attention/full-attention decoder, 64 layers, hidden_size 5120)
over ProcessGroupQUIC - the "larger real model" proof from the
deliverable report's recommended next steps, one level up from
real_llm_pipeline_rank.py's Qwen2.5-0.5B.

4 stages, not 2: each of the 2 real machines has 2 T4s, so each stage
gets 16 layers on its OWN, single, real GPU - trivially fits a 15GB T4
(16/64 layers in 4-bit is ~3.4GB). The OTHER 48 layers this rank
doesn't need still have to be pointed SOMEWHERE in device_map (a
transformers requirement); they land on the meta device with zero real
memory ever materialized (confirmed via a direct load test, not
assumed) via a "cpu" device_map entry + llm_int8_enable_fp32_cpu_offload
- no actual CPU compute or memory cost, just a way of saying "not on my
GPU" that this transformers version accepts. Topology: rank0 (local GPU0) -> rank1 (local GPU1,
same machine - the hop between rank0 and rank1 is a REAL quic_dist
connection too, just over loopback) -> rank2 (2nd machine GPU0, the
one REAL cross-machine hop) -> rank3 (2nd machine GPU1). Exercises
quic_dist's N>2 support at world_size=4 with a real, non-toy workload,
on top of everything test_n_gt_2.py/n3_cross_machine_rank.py already
proved about the topology/protocol logic itself.

AutoModelForCausalLM auto-selects Qwen3_5ForCausalLM (text-only, no
vision tower) for this config - confirmed via a meta-device skeleton
inspection, not assumed: 26.9B params, clean top-level
model.embed_tokens / model.layers / model.norm / lm_head, matching the
existing pipeline-split pattern exactly.

LoRA targets gate_proj/up_proj (MLP) - present identically on every
layer regardless of block_type (linear_attention vs full_attention),
unlike the attention projections which have different names/shapes
between the two block types (self_attn.{q,k,v,o}_proj vs
linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}) -
sidesteps needing separate handling for the two attention mechanisms.

Dataset: a real, small, fixed slice of tatsu-lab/alpaca (a real,
widely-used instruction-tuning dataset - not synthetic), same text
format all ranks reconstruct identically and locally, so no raw text
crosses the network, only the pipeline-boundary activation/gradient -
same convention as real_llm_pipeline_rank.py.

Usage: python3 qwen38_27b_pipeline_rank.py <rank> <signaling_url> [job_id]
(rank in 0..3; local GPU index used is rank % 2)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn

import quic_dist
import torch.distributed as dist

MODEL_PATH = "/models/qwen38_27b"
NUM_LAYERS = 64
WORLD_SIZE = 4
# Uneven split, not 16/16/16/16 - rank3 (last stage) hit CUDA OOM TWICE at an
# even split, first in the linear-attention reference forward, then (after
# shrinking SEQ_LEN) at lm_head itself - both failures within ~25MiB of the
# 14.56GiB T4 limit. lm_head (5120 x 248320, kept unquantized/bf16 by
# bitsandbytes' default skip-list) plus norm cost rank3 ~2.5GB extra fixed
# weight on top of whatever decoder layers it holds, that no other rank
# pays - so it gets fewer layers to compensate. Confirmed via two real OOM
# tracebacks, not guessed.
STAGE_LAYER_COUNTS = [17, 17, 18, 12]  # sums to 64; rank3 deliberately smallest
NUM_EXAMPLES = 64  # a small, fixed, real slice - proof of genuine training, not full fine-tuning
SEQ_LEN = 48  # also reduced from 96 - the linear-attention reference implementation's
              # chunked forward (no fused kernel installed - causal_conv1d/fla aren't
              # present in this venv) is memory-hungry and scales with sequence length.
              # pipeline. Confirmed via a real OOM traceback, not guessed.
BATCH = 1
EPOCHS = 3
LR = 1e-4


def stage_range(rank: int) -> range:
    start = sum(STAGE_LAYER_COUNTS[:rank])
    return range(start, start + STAGE_LAYER_COUNTS[rank])


def build_dataset(tokenizer):
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split=f"train[:{NUM_EXAMPLES}]")
    block = SEQ_LEN + 1
    all_ids = []
    for ex in ds:
        ids = tokenizer(ex["text"], truncation=True, max_length=block, padding="max_length")["input_ids"]
        all_ids.append(ids)
    ids_t = torch.tensor(all_ids, dtype=torch.long)  # (NUM_EXAMPLES, block)
    n_batches = ids_t.shape[0] // BATCH
    ids_t = ids_t[: n_batches * BATCH].view(n_batches, BATCH, block)
    return ids_t


def build_device_map(rank: int, local_gpu: int) -> dict:
    # "cpu" here, not "meta" - transformers doesn't accept "meta" as an
    # explicit device_map value. Combined with
    # llm_int8_enable_fp32_cpu_offload=True (required below whenever any
    # module is "cpu"/"disk"-mapped), "cpu"-mapped modules end up on the
    # meta device with ZERO real memory ever materialized for them -
    # confirmed via a direct load test, not assumed (see module
    # docstring's predecessor script's own note on this). We never call
    # into these layers, so their exact placement doesn't matter beyond
    # "not on this rank's GPU".
    my_layers = set(stage_range(rank))
    device_map = {}
    for i in range(NUM_LAYERS):
        device_map[f"model.layers.{i}"] = local_gpu if i in my_layers else "cpu"
    device_map["model.embed_tokens"] = local_gpu if rank == 0 else "cpu"
    device_map["model.norm"] = local_gpu if rank == WORLD_SIZE - 1 else "cpu"
    device_map["model.rotary_emb"] = local_gpu
    device_map["lm_head"] = local_gpu if rank == WORLD_SIZE - 1 else "cpu"
    return device_map


def build_model(rank: int, local_gpu: int):
    import peft.tuners.lora.torchao as torchao_mod

    torchao_mod.is_torchao_available = lambda: False  # see real_llm_pipeline_rank.py's docstring - same env bug

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from transformers.utils import logging as hflog
    from peft import LoraConfig, get_peft_model

    hflog.disable_progress_bar()
    hflog.set_verbosity_error()

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, quantization_config=bnb_cfg, device_map=build_device_map(rank, local_gpu)
    )

    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["gate_proj", "up_proj"], lora_dropout=0.0)
    peft_model = get_peft_model(model, lora_cfg)
    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"[rank {rank}] trainable params: {n_trainable}, layers {list(stage_range(rank))}", flush=True)
    return peft_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rank", type=int)
    parser.add_argument("signaling_url")
    parser.add_argument("job_id", nargs="?", default="qwen38_27b")
    args = parser.parse_args()
    rank = args.rank
    local_gpu = rank % 2
    device = torch.device(f"cuda:{local_gpu}")
    prev_rank = rank - 1 if rank > 0 else None
    next_rank = rank + 1 if rank < WORLD_SIZE - 1 else None
    is_first = rank == 0
    is_last = rank == WORLD_SIZE - 1

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quic_dist.init_process_group(
        signaling_url=args.signaling_url, rank=rank, world_size=WORLD_SIZE, job_id=args.job_id,
        timeout=timedelta(seconds=1800),
    )
    print(f"[rank {rank}] process group ready (local GPU {local_gpu})", flush=True)

    peft_model = build_model(rank, local_gpu)
    inner = peft_model.base_model.model.model  # Qwen3_5TextModel-equivalent: embed_tokens, layers, norm, rotary_emb
    lm_head = peft_model.base_model.model.lm_head
    hidden_size = inner.config.hidden_size

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR)

    batches = build_dataset(tokenizer)
    n_steps = batches.shape[0]
    total_steps = n_steps * EPOCHS
    print(f"[rank {rank}] {EPOCHS} epochs x {n_steps} steps/epoch = {total_steps} steps", flush=True)

    losses = []
    t_start = time.monotonic()
    step_counter = 0
    my_layers = inner.layers[stage_range(rank).start : stage_range(rank).stop]

    for epoch in range(EPOCHS):
        for b in range(n_steps):
            step_t0 = time.monotonic()
            tag = step_counter % 8
            step_counter += 1
            block = batches[b]
            optimizer.zero_grad()
            position_ids = torch.arange(SEQ_LEN, device=device).unsqueeze(0).expand(BATCH, -1)

            if is_first:
                input_ids = block[:, :-1].to(device)
                hidden = inner.embed_tokens(input_ids)
            else:
                recv_buf = torch.zeros(BATCH, SEQ_LEN, hidden_size, dtype=torch.bfloat16)
                dist.recv(recv_buf, src=prev_rank, tag=tag)
                hidden = recv_buf.to(device).requires_grad_(True)

            pos_emb = inner.rotary_emb(hidden, position_ids)
            out = hidden
            for layer in my_layers:
                out = layer(out, attention_mask=None, position_ids=position_ids, position_embeddings=pos_emb)

            if is_last:
                labels = block[:, 1:].to(device)
                out = inner.norm(out)
                logits = lm_head(out)
                loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1))
                loss.backward()
                losses.append(loss.item())
            else:
                dist.send(out.detach().to(torch.bfloat16).cpu(), dst=next_rank, tag=tag)
                grad = torch.zeros(BATCH, SEQ_LEN, hidden_size, dtype=torch.bfloat16)
                dist.recv(grad, src=next_rank, tag=tag)
                out.backward(grad.to(device))

            if not is_first:
                dist.send(hidden.grad.detach().to(torch.bfloat16).cpu(), dst=prev_rank, tag=tag)

            optimizer.step()
            if step_counter <= 3 or step_counter % 16 == 0:
                print(f"[rank {rank}] step {step_counter}/{total_steps} took {time.monotonic()-step_t0:.1f}s", flush=True)

        elapsed = time.monotonic() - t_start
        if is_last:
            print(f"[rank {rank}] epoch {epoch}/{EPOCHS} loss={losses[-1]:.4f} elapsed={elapsed:.1f}s", flush=True)
        else:
            print(f"[rank {rank}] epoch {epoch}/{EPOCHS} elapsed={elapsed:.1f}s", flush=True)

    total_elapsed = time.monotonic() - t_start
    print(f"[rank {rank}] training DONE in {total_elapsed:.1f}s ({total_steps} steps)", flush=True)

    if is_last:
        first_n = sum(losses[:n_steps]) / n_steps
        last_n = sum(losses[-n_steps:]) / n_steps
        print(f"[rank {rank}] loss avg epoch 0:  {first_n:.4f}", flush=True)
        print(f"[rank {rank}] loss avg last epoch: {last_n:.4f}", flush=True)
        print(f"[rank {rank}] all losses: {losses}", flush=True)

    dist.barrier()
    pg = dist.distributed_c10d._get_default_group()
    dist.destroy_process_group()
    pg.close_connections()
    print(f"[rank {rank}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
