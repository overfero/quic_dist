"""Direct proof that ProcessGroupQUIC/quic_dist.tensor are genuinely
shape- and rank-agnostic, not implicitly assuming a (batch, seq, hidden)
3D text-transformer activation shape anywhere. Sends real vision-model-
shaped tensors - a 4D CNN/ViT-style (batch, channels, H, W) activation
and a 5D video/temporal-ViT-style (batch, frames, channels, H, W) one -
cross-machine, and confirms a byte-exact round trip. Cheaper and more
direct than a full vision-model training run for proving this specific
claim (tensor.py's serialize/deserialize never reference a hardcoded
axis count - see _MAX_NDIM=16 and its generic shape-packing format).

Usage: python3 vision_shape_rank.py <rank> <signaling_url> [job_id]
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # parent of quic_dist/, works regardless of clone location as long as the repo dir is named quic_dist

import torch
import torch.distributed as dist

import quic_dist


def main() -> None:
    rank = int(sys.argv[1])
    signaling_url = sys.argv[2]
    job_id = sys.argv[3] if len(sys.argv) > 3 else "vision_shape"

    quic_dist.init_process_group(
        signaling_url=signaling_url, rank=rank, world_size=2, job_id=job_id, timeout=timedelta(seconds=60)
    )
    print(f"[rank {rank}] process group ready", flush=True)

    torch.manual_seed(42)
    cnn_shaped = torch.randn(4, 3, 64, 64)  # (batch, channels, H, W) - e.g. a ViT patch-embed / CNN activation
    video_shaped = torch.randn(2, 8, 16, 8, 8)  # (batch, frames, channels, h, w) - temporal/video-style, 5D
    scalar_shaped = torch.tensor(3.14159)  # 0-dim, an edge case worth covering too

    if rank == 0:
        dist.send(cnn_shaped, dst=1, tag=0)
        dist.send(video_shaped, dst=1, tag=1)
        dist.send(scalar_shaped, dst=1, tag=2)
        print(f"[rank 0] sent 4D, 5D, 0D tensors", flush=True)
    else:
        recv_4d = torch.zeros(4, 3, 64, 64)
        dist.recv(recv_4d, src=0, tag=0)
        assert torch.equal(recv_4d, cnn_shaped), "4D (vision-shaped) tensor corrupted in transit!"
        print(f"[rank 1] 4D vision-shaped tensor round-trip CORRECT: shape={tuple(recv_4d.shape)}", flush=True)

        recv_5d = torch.zeros(2, 8, 16, 8, 8)
        dist.recv(recv_5d, src=0, tag=1)
        assert torch.equal(recv_5d, video_shaped), "5D (video-shaped) tensor corrupted in transit!"
        print(f"[rank 1] 5D video-shaped tensor round-trip CORRECT: shape={tuple(recv_5d.shape)}", flush=True)

        recv_0d = torch.zeros(())
        dist.recv(recv_0d, src=0, tag=2)
        assert torch.equal(recv_0d, scalar_shaped), "0D scalar tensor corrupted in transit!"
        print(f"[rank 1] 0D scalar tensor round-trip CORRECT: value={recv_0d.item()}", flush=True)

        print("[rank 1] ALL SHAPES CORRECT - quic_dist is genuinely shape/rank-agnostic", flush=True)

    dist.barrier()
    pg = dist.distributed_c10d._get_default_group()
    dist.destroy_process_group()
    pg.close_connections()
    print(f"[rank {rank}] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
