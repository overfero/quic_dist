"""Real 2-stage pipeline-parallel fine-tuning of a real pretrained LLM
(Qwen2.5-0.5B) over one epoch of a real text dataset, cross-machine, over
ProcessGroupQUIC - the concrete "is this actually possible" proof asked
for, one level up from lora_pipeline_rank.py's hand-rolled toy model.

Supports --mode lora (standard LoRA, fp16 base) and --mode qlora (4-bit
NF4 base via bitsandbytes + prepare_model_for_kbit_training + LoRA on
top) - both go through the exact same quic_dist send/recv/isend/irecv
calls, because quic_dist has no LoRA/QLoRA-specific code anywhere: it
moves plain torch.Tensor objects tagged by an int, and has no idea
whether the sender used LoRA, QLoRA, full fine-tuning, or something else
entirely. That's the point being demonstrated here, not just asserted.

Dataset: tiny_shakespeare (plain text, downloaded once, synced to both
machines identically) - each rank tokenizes the SAME local copy with the
SAME deterministic chunking, so no raw text/token IDs cross the network,
only the activation/gradient tensors at the pipeline boundary. Real
causal masking (verified separately: attention_mask=None triggers a real
causal default here, confirmed via a position-0-unaffected-by-position-7
probe), real next-token cross-entropy loss.

Works around the same real peft/torchao environment bug documented in
lora_pipeline_rank.py's docstring - confirmed here to affect ALL LoRA
target types in this environment (plain nn.Linear included, not just
Conv1D as first suspected), so the workaround is applied unconditionally.

Usage: python3 real_llm_pipeline_rank.py <rank> <signaling_url> [--mode lora|qlora] [job_id]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import time
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # parent of quic_dist/, works regardless of clone location as long as the repo dir is named quic_dist

import torch
import torch.nn as nn

import quic_dist
import torch.distributed as dist

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATA_PATH = str(Path(__file__).resolve().parent / "data" / "tinyshakespeare.txt")
DATA_CHAR_LIMIT = 300_000  # ~30% of the corpus - a real but quick "one epoch"
SEQ_LEN = 128
BATCH = 4
SPLIT_LAYER = 12  # stage0: layers[:12] ; stage1: layers[12:]
LR = 1e-4
LOG_EVERY = 20


def build_dataset(tokenizer):
    with open(DATA_PATH, "r") as f:
        text = f.read()[:DATA_CHAR_LIMIT]
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
    block = SEQ_LEN + 1
    n_blocks = len(ids) // block
    ids = ids[: n_blocks * block].view(n_blocks, block)
    n_batches = n_blocks // BATCH
    ids = ids[: n_batches * BATCH].view(n_batches, BATCH, block)
    return ids  # (n_batches, BATCH, SEQ_LEN+1)


def build_model(mode: str, device: torch.device):
    import peft.tuners.lora.torchao as torchao_mod

    torchao_mod.is_torchao_available = lambda: False  # see module docstring

    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from transformers.utils import logging as hflog
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    hflog.disable_progress_bar()
    hflog.set_verbosity_error()

    if mode == "qlora":
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_cfg, device_map={"": device.index})
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16).to(device)

    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], lora_dropout=0.0)
    peft_model = get_peft_model(model, lora_cfg)
    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"[mode={mode}] trainable params: {n_trainable}", flush=True)
    return peft_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rank", type=int)
    parser.add_argument("signaling_url")
    parser.add_argument("job_id", nargs="?", default="real_llm")
    parser.add_argument("--mode", choices=["lora", "qlora"], default="lora")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    quic_dist.init_process_group(
        signaling_url=args.signaling_url, rank=args.rank, world_size=2, job_id=args.job_id,
        timeout=timedelta(seconds=120),
    )
    print(f"[rank {args.rank}] process group ready", flush=True)

    peft_model = build_model(args.mode, device)
    inner = peft_model.base_model.model.model  # Qwen2Model (embed_tokens, layers, norm, rotary_emb)
    lm_head = peft_model.base_model.model.lm_head

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR)

    batches = build_dataset(tokenizer)
    n_steps = batches.shape[0]
    print(f"[rank {args.rank}] one epoch = {n_steps} steps ({n_steps * BATCH * SEQ_LEN} tokens)", flush=True)

    losses = []
    t_start = time.monotonic()
    for step in range(n_steps):
        block = batches[step].to(device)  # (BATCH, SEQ_LEN+1)
        input_ids = block[:, :-1]
        labels = block[:, 1:]

        optimizer.zero_grad()
        position_ids = torch.arange(SEQ_LEN, device=device).unsqueeze(0).expand(BATCH, -1)

        if args.rank == 0:
            hidden = inner.embed_tokens(input_ids)
            pos_emb = inner.rotary_emb(hidden, position_ids)
            for layer in inner.layers[:SPLIT_LAYER]:
                hidden = layer(hidden, attention_mask=None, position_ids=position_ids, position_embeddings=pos_emb)
            dist.send(hidden.detach().to(torch.float16).cpu(), dst=1, tag=step % 8)

            grad = torch.zeros(BATCH, SEQ_LEN, inner.config.hidden_size, dtype=torch.float16)
            dist.recv(grad, src=1, tag=step % 8)
            hidden.backward(grad.to(device))
            optimizer.step()
        else:
            recv_buf = torch.zeros(BATCH, SEQ_LEN, inner.config.hidden_size, dtype=torch.float16)
            dist.recv(recv_buf, src=0, tag=step % 8)
            hidden = recv_buf.to(device).requires_grad_(True)

            pos_emb = inner.rotary_emb(hidden, position_ids)
            out = hidden
            for layer in inner.layers[SPLIT_LAYER:]:
                out = layer(out, attention_mask=None, position_ids=position_ids, position_embeddings=pos_emb)
            out = inner.norm(out)
            logits = lm_head(out)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1))
            loss.backward()
            dist.send(hidden.grad.detach().to(torch.float16).cpu(), dst=0, tag=step % 8)
            optimizer.step()
            losses.append(loss.item())

        if step % LOG_EVERY == 0 or step == n_steps - 1:
            elapsed = time.monotonic() - t_start
            if args.rank == 1:
                print(f"[rank 1] step {step}/{n_steps} loss={losses[-1]:.4f} elapsed={elapsed:.1f}s", flush=True)
            else:
                print(f"[rank 0] step {step}/{n_steps} elapsed={elapsed:.1f}s", flush=True)

    total_elapsed = time.monotonic() - t_start
    print(f"[rank {args.rank}] one epoch DONE in {total_elapsed:.1f}s ({n_steps} steps)", flush=True)

    if args.rank == 1:
        first_10_avg = sum(losses[:10]) / len(losses[:10])
        last_10_avg = sum(losses[-10:]) / len(losses[-10:])
        print(f"[rank 1] loss avg first 10 steps: {first_10_avg:.4f}", flush=True)
        print(f"[rank 1] loss avg last 10 steps:  {last_10_avg:.4f}", flush=True)
        print(f"[rank 1] all losses: {losses}", flush=True)

    dist.barrier()
    pg = dist.distributed_c10d._get_default_group()
    dist.destroy_process_group()
    pg.close_connections()
    print(f"[rank {args.rank}] ALL DONE, mode={args.mode}", flush=True)


if __name__ == "__main__":
    main()
