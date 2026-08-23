"""Real 2-stage pipeline-parallel LoRA fine-tuning of a real pretrained
vision model (ViT-Tiny/patch16/224, WinKawaks/vit-tiny-patch16-224) over
ProcessGroupQUIC, cross-machine - the vision-model analog of
real_llm_pipeline_rank.py. Closes the "real vision/multimodal model
forward+backward" gap noted in the deliverable report: previously only
the transport's shape-agnosticism (4D/5D/0D tensor round-trips) and the
LLM training proof existed separately: this combines both by running a
real conv-patch-embedding + attention + LayerNorm architecture (not a
text transformer) through the exact same send/recv/isend/irecv calls,
with real LoRA adapters on the attention q/k/v projections.

Data: a small FIXED synthetic image/label set (deterministically seeded,
same on both ranks - no image ever crosses the network, only the
pipeline-boundary activation/gradient at [CLS]+patch tokens). The point
is not classification accuracy on real photos - it's proving genuine
weight updates flow correctly through this transport for a fundamentally
different (non-causal, non-text) architecture: real loss decrease over
several epochs of memorizing a small fixed set is the correctness
signal, same rationale as tiny_shakespeare's role in the LLM proof.

Usage: python3 vision_pipeline_rank.py <rank> <signaling_url> [job_id]
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

MODEL_NAME = "WinKawaks/vit-tiny-patch16-224"
SPLIT_LAYER = 6  # stage0: encoder.layer[:6] ; stage1: encoder.layer[6:] + layernorm + classifier
NUM_SAMPLES = 32
NUM_CLASSES_USED = 8  # a small fixed label set out of the real 1000-way ImageNet head
BATCH = 4
EPOCHS = 15
LR = 5e-4
LOG_EVERY = 4


def build_synthetic_dataset(device: torch.device):
    g = torch.Generator().manual_seed(1234)
    images = torch.randn(NUM_SAMPLES, 3, 224, 224, generator=g)
    labels = torch.randint(0, NUM_CLASSES_USED, (NUM_SAMPLES,), generator=g)
    n_batches = NUM_SAMPLES // BATCH
    images = images[: n_batches * BATCH].view(n_batches, BATCH, 3, 224, 224)
    labels = labels[: n_batches * BATCH].view(n_batches, BATCH)
    return images, labels


def build_model(device: torch.device):
    import peft.tuners.lora.torchao as torchao_mod

    torchao_mod.is_torchao_available = lambda: False  # see real_llm_pipeline_rank.py's docstring - same env bug

    from transformers import ViTForImageClassification
    from transformers.utils import logging as hflog
    from peft import LoraConfig, get_peft_model

    hflog.disable_progress_bar()
    hflog.set_verbosity_error()

    model = ViTForImageClassification.from_pretrained(MODEL_NAME, dtype=torch.float32).to(device)
    lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"], lora_dropout=0.0)
    peft_model = get_peft_model(model, lora_cfg)
    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    print(f"trainable params: {n_trainable}", flush=True)
    return peft_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rank", type=int)
    parser.add_argument("signaling_url")
    parser.add_argument("job_id", nargs="?", default="vision_pipeline")
    args = parser.parse_args()

    device = torch.device("cuda:0")

    quic_dist.init_process_group(
        signaling_url=args.signaling_url, rank=args.rank, world_size=2, job_id=args.job_id,
        timeout=timedelta(seconds=120),
    )
    print(f"[rank {args.rank}] process group ready", flush=True)

    peft_model = build_model(device)
    vit = peft_model.base_model.model.vit  # ViTModel: embeddings, encoder.layer, layernorm
    classifier = peft_model.base_model.model.classifier

    images, labels = build_synthetic_dataset(device)
    n_steps_per_epoch = images.shape[0]
    total_steps = n_steps_per_epoch * EPOCHS
    print(f"[rank {args.rank}] {EPOCHS} epochs x {n_steps_per_epoch} steps/epoch = {total_steps} steps", flush=True)

    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=LR)

    losses = []
    t_start = time.monotonic()
    step_counter = 0
    for epoch in range(EPOCHS):
        for b in range(n_steps_per_epoch):
            tag = step_counter % 8
            step_counter += 1
            optimizer.zero_grad()

            if args.rank == 0:
                pixel_values = images[b].to(device)
                hidden = vit.embeddings(pixel_values)
                for layer in vit.encoder.layer[:SPLIT_LAYER]:
                    hidden = layer(hidden)
                dist.send(hidden.detach().cpu(), dst=1, tag=tag)

                grad = torch.zeros_like(hidden, device="cpu")
                dist.recv(grad, src=1, tag=tag)
                hidden.backward(grad.to(device))
                optimizer.step()
            else:
                target = labels[b].to(device)
                recv_buf = torch.zeros(BATCH, 197, vit.config.hidden_size)
                dist.recv(recv_buf, src=0, tag=tag)
                hidden = recv_buf.to(device).requires_grad_(True)

                out = hidden
                for layer in vit.encoder.layer[SPLIT_LAYER:]:
                    out = layer(out)
                out = vit.layernorm(out)
                cls_token = out[:, 0]
                logits = classifier(cls_token)
                loss = nn.functional.cross_entropy(logits, target)
                loss.backward()
                dist.send(hidden.grad.detach().cpu(), dst=0, tag=tag)
                optimizer.step()
                losses.append(loss.item())

        if epoch % 2 == 0 or epoch == EPOCHS - 1:
            elapsed = time.monotonic() - t_start
            if args.rank == 1:
                print(f"[rank 1] epoch {epoch}/{EPOCHS} loss={losses[-1]:.4f} elapsed={elapsed:.1f}s", flush=True)
            else:
                print(f"[rank 0] epoch {epoch}/{EPOCHS} elapsed={elapsed:.1f}s", flush=True)

    total_elapsed = time.monotonic() - t_start
    print(f"[rank {args.rank}] training DONE in {total_elapsed:.1f}s ({total_steps} steps)", flush=True)

    if args.rank == 1:
        first_n = sum(losses[:n_steps_per_epoch]) / n_steps_per_epoch
        last_n = sum(losses[-n_steps_per_epoch:]) / n_steps_per_epoch
        print(f"[rank 1] loss avg epoch 0:  {first_n:.4f}", flush=True)
        print(f"[rank 1] loss avg last epoch: {last_n:.4f}", flush=True)
        print(f"[rank 1] all losses: {losses}", flush=True)

    dist.barrier()
    pg = dist.distributed_c10d._get_default_group()
    dist.destroy_process_group()
    pg.close_connections()
    print(f"[rank {args.rank}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
