"""Real cross-machine correctness check for ProcessGroupQUIC: run this on
two genuinely different machines (not loopback), each with a different
rank, pointed at the same public signaling URL - the first real test of
this project's NAT hole-punch actually crossing a real network boundary
for `quic_dist`, rather than two processes on the same host.

Usage: python3 cross_machine_rank.py <rank> <signaling_url> [job_id]
"""
from __future__ import annotations

import sys
from pathlib import Path
import time
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # parent of quic_dist/, works regardless of clone location as long as the repo dir is named quic_dist

import torch
import torch.distributed as dist

import quic_dist


def main() -> None:
    rank = int(sys.argv[1])
    signaling_url = sys.argv[2]
    job_id = sys.argv[3] if len(sys.argv) > 3 else "cross_machine"
    other = 1 - rank

    print(f"[rank {rank}] init_process_group...", flush=True)
    t0 = time.monotonic()
    quic_dist.init_process_group(
        signaling_url=signaling_url, rank=rank, world_size=2, job_id=job_id, timeout=timedelta(seconds=120)
    )
    print(f"[rank {rank}] init done in {time.monotonic() - t0:.2f}s", flush=True)

    # send/recv roundtrip
    if rank == 0:
        tensor = torch.arange(2000, dtype=torch.float32)
        t0 = time.monotonic()
        dist.send(tensor, dst=1, tag=0)
        print(f"[rank 0] send done in {time.monotonic() - t0:.3f}s", flush=True)

        out = torch.zeros(2000, dtype=torch.float32)
        t0 = time.monotonic()
        dist.recv(out, src=1, tag=1)
        print(f"[rank 0] recv done in {time.monotonic() - t0:.3f}s", flush=True)
        assert torch.equal(out, tensor * 2), "cross-machine reply mismatch!"
        print("[rank 0] send/recv roundtrip CORRECT", flush=True)
    else:
        out = torch.zeros(2000, dtype=torch.float32)
        t0 = time.monotonic()
        dist.recv(out, src=0, tag=0)
        print(f"[rank 1] recv done in {time.monotonic() - t0:.3f}s", flush=True)
        expected = torch.arange(2000, dtype=torch.float32)
        assert torch.equal(out, expected), "cross-machine forward mismatch!"

        t0 = time.monotonic()
        dist.send(out * 2, dst=0, tag=1)
        print(f"[rank 1] send done in {time.monotonic() - t0:.3f}s", flush=True)

    # isend/irecv, several tags in flight
    if rank == 0:
        tensors = [torch.full((500,), float(i)) for i in range(4)]
        t0 = time.monotonic()
        works = [dist.isend(tensors[i], dst=1, tag=10 + i) for i in range(4)]
        for w in works:
            w.wait()
        print(f"[rank 0] isend x4 done in {time.monotonic() - t0:.3f}s", flush=True)
    else:
        outs = [torch.zeros(500) for _ in range(4)]
        t0 = time.monotonic()
        works = [dist.irecv(outs[i], src=0, tag=10 + i) for i in range(4)]
        for w in works:
            w.wait()
        print(f"[rank 1] irecv x4 done in {time.monotonic() - t0:.3f}s", flush=True)
        for i, out in enumerate(outs):
            assert torch.equal(out, torch.full((500,), float(i))), f"tag {10+i} corrupted"
        print("[rank 1] multi-tag isend/irecv CORRECT", flush=True)

    # barrier
    t0 = time.monotonic()
    dist.barrier()
    print(f"[rank {rank}] barrier done in {time.monotonic() - t0:.3f}s", flush=True)

    # 10MB throughput sample, one direction
    if rank == 0:
        big = torch.randn(10 * 1024 * 1024 // 4, dtype=torch.float32)
        t0 = time.monotonic()
        dist.send(big, dst=1, tag=99)
        elapsed = time.monotonic() - t0
        mbps = (10 * 8) / elapsed
        print(f"[rank 0] sent 10MB in {elapsed:.3f}s ({mbps:.1f} Mbps)", flush=True)
    else:
        big = torch.zeros(10 * 1024 * 1024 // 4, dtype=torch.float32)
        t0 = time.monotonic()
        dist.recv(big, src=0, tag=99)
        elapsed = time.monotonic() - t0
        mbps = (10 * 8) / elapsed
        print(f"[rank 1] received 10MB in {elapsed:.3f}s ({mbps:.1f} Mbps)", flush=True)

    dist.barrier()
    pg = dist.distributed_c10d._get_default_group()
    dist.destroy_process_group()
    pg.close_connections()
    print(f"[rank {rank}] ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
