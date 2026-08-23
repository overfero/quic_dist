"""Real 2-stage pipeline-parallel LoRA forward+backward step, cross-machine,
over ProcessGroupQUIC. Each rank owns half of a small transformer's layers
on its own GPU; activations flow rank0->rank1, gradients flow rank1->rank0,
each rank runs backward()/optimizer.step() on only its own LoRA params -
the actual shape of the workload quic_dist was built for (see
parallel_training_prompt.md).

A hand-rolled model (nn.TransformerEncoderLayer + nn.Linear LoRA targets),
not a real HF checkpoint - deliberately small/self-contained (no download,
no risk from an unfamiliar model's internal API), the correctness question
being tested here is quic_dist's real tensor+gradient exchange under real
autograd, not model quality. Works around a real environment bug (peft
0.19.1 unconditionally probes torchao during Conv1D LoRA dispatch, and
torchao 0.10.0 fails that probe by raising instead of returning False) by
using nn.Linear LoRA targets, which never hit that dispatch path at all.

Usage: python3 lora_pipeline_rank.py <rank> <signaling_url> [job_id]
"""
from __future__ import annotations

import sys
from pathlib import Path
import time
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # parent of quic_dist/, works regardless of clone location as long as the repo dir is named quic_dist

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

import quic_dist
import torch.distributed as dist

VOCAB = 200
D_MODEL = 64
NHEAD = 4
DIM_FF = 128
NUM_LAYERS = 4
BATCH = 2
SEQLEN = 12
NUM_STEPS = 3


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    D_MODEL, NHEAD, DIM_FF, batch_first=True, dropout=0.0
                )
                for _ in range(NUM_LAYERS)
            ]
        )
        self.lm_head = nn.Linear(D_MODEL, VOCAB)


def build_model(device: torch.device):
    torch.manual_seed(0)
    model = TinyTransformer()
    lora_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["linear1", "linear2"], lora_dropout=0.0)
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.to(device)
    return peft_model


def main() -> None:
    import peft.tuners.lora.torchao as torchao_mod

    torchao_mod.is_torchao_available = lambda: False  # see module docstring

    rank = int(sys.argv[1])
    signaling_url = sys.argv[2]
    job_id = sys.argv[3] if len(sys.argv) > 3 else "lora_pipeline"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[rank {rank}] device={device}", flush=True)

    quic_dist.init_process_group(
        signaling_url=signaling_url, rank=rank, world_size=2, job_id=job_id, timeout=timedelta(seconds=120)
    )
    print(f"[rank {rank}] process group ready", flush=True)

    peft_model = build_model(device)
    inner = peft_model.base_model.model  # TinyTransformer, with LoRA-wrapped linear1/linear2
    causal_mask = nn.Transformer.generate_square_subsequent_mask(SEQLEN).to(device)

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=0.1)
    n_trainable = sum(p.numel() for p in trainable)
    print(f"[rank {rank}] trainable LoRA params: {n_trainable}", flush=True)

    split = NUM_LAYERS // 2  # stage0: layers[:split] ; stage1: layers[split:]

    losses = []
    for step in range(NUM_STEPS):
        torch.manual_seed(1000 + step)  # same input_ids on both ranks for this microbatch
        input_ids = torch.randint(0, VOCAB, (BATCH, SEQLEN))
        labels = torch.randint(0, VOCAB, (BATCH, SEQLEN))

        optimizer.zero_grad()
        t0 = time.monotonic()

        if rank == 0:
            hidden = inner.embed(input_ids.to(device))
            for layer in inner.layers[:split]:
                hidden = layer(hidden, src_mask=causal_mask)
            dist.send(hidden.detach().cpu(), dst=1, tag=step)

            grad = torch.zeros(BATCH, SEQLEN, D_MODEL)
            dist.recv(grad, src=1, tag=step)
            hidden.backward(grad.to(device))
            optimizer.step()
            elapsed = time.monotonic() - t0
            print(f"[rank 0] step {step}: fwd+bwd round-trip in {elapsed:.3f}s", flush=True)
        else:
            recv_buf = torch.zeros(BATCH, SEQLEN, D_MODEL)
            dist.recv(recv_buf, src=0, tag=step)
            hidden = recv_buf.to(device).requires_grad_(True)

            out = hidden
            for layer in inner.layers[split:]:
                out = layer(out, src_mask=causal_mask)
            logits = peft_model.base_model.model.lm_head(out)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB), labels.to(device).reshape(-1)
            )
            loss.backward()
            dist.send(hidden.grad.detach().cpu(), dst=0, tag=step)
            optimizer.step()

            losses.append(loss.item())
            elapsed = time.monotonic() - t0
            print(f"[rank 1] step {step}: loss={loss.item():.4f} in {elapsed:.3f}s", flush=True)

        dist.barrier()

    if rank == 1:
        print(f"[rank 1] losses across steps: {losses}", flush=True)
        assert all(torch.isfinite(torch.tensor(x)) for x in losses), "non-finite loss - real correctness failure"
        print("[rank 1] all losses finite - CORRECT", flush=True)

    dist.barrier()
    pg = dist.distributed_c10d._get_default_group()
    dist.destroy_process_group()
    pg.close_connections()
    print(f"[rank {rank}] ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
