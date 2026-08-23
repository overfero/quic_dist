"""Throughput + latency benchmark for ProcessGroupQUIC, through the
standard torch.distributed API (send/recv/isend/irecv), not the raw Rust
driver - this measures what a real training loop actually experiences,
including tensor serialization overhead and the per-tag dispatch thread.

Run directly: `python3 bench_quic_dist.py` (spawns its own signaling
server + 2 worker processes, loopback, matching this project's
established benchmark convention - real subprocesses, real sockets).
"""
from __future__ import annotations

import multiprocessing as mp
import statistics
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # parent of quic_dist/, for `import quic_dist`
sys.path.insert(0, str(Path(__file__).resolve().parent))  # this directory, for `_helpers`

import torch

MB = 1024 * 1024
PAYLOAD_SIZES_MB = [1, 10, 100]
IN_FLIGHT_COUNTS = [1, 4, 8, 16]
IN_FLIGHT_PAYLOAD_MB = 4  # per-message size used for the in-flight sweep
WARMUP_ROUNDS = 2
TIMED_ROUNDS = 5
MP_CTX = mp.get_context("fork")


def _throughput_sender(barrier, signaling_url: str, size_mb: int, result_queue):
    import quic_dist
    import torch.distributed as dist

    quic_dist.init_process_group(
        signaling_url=signaling_url, rank=0, world_size=2, job_id=f"bench_tp_{size_mb}", timeout=timedelta(seconds=60)
    )
    tensor = torch.randn(size_mb * MB // 4, dtype=torch.float32)  # float32 = 4 bytes/elem
    barrier.wait()
    for _ in range(WARMUP_ROUNDS):
        dist.send(tensor, dst=1, tag=0)
    timings = []
    for _ in range(TIMED_ROUNDS):
        t0 = time.perf_counter()
        dist.send(tensor, dst=1, tag=0)
        timings.append(time.perf_counter() - t0)
    dist.barrier()
    result_queue.put({"rank": 0, "timings": timings})

    pg = dist.distributed_c10d._get_default_group()
    dist.destroy_process_group()
    pg.close_connections()


def _throughput_receiver(barrier, signaling_url: str, size_mb: int, result_queue):
    import quic_dist
    import torch.distributed as dist

    quic_dist.init_process_group(
        signaling_url=signaling_url, rank=1, world_size=2, job_id=f"bench_tp_{size_mb}", timeout=timedelta(seconds=60)
    )
    tensor = torch.zeros(size_mb * MB // 4, dtype=torch.float32)
    barrier.wait()
    for _ in range(WARMUP_ROUNDS):
        dist.recv(tensor, src=0, tag=0)
    for _ in range(TIMED_ROUNDS):
        dist.recv(tensor, src=0, tag=0)
    dist.barrier()
    result_queue.put({"rank": 1})

    pg = dist.distributed_c10d._get_default_group()
    dist.destroy_process_group()
    pg.close_connections()


def bench_throughput(signaling_url: str, size_mb: int) -> dict:
    barrier = MP_CTX.Barrier(2)
    result_queue = MP_CTX.Queue()
    procs = [
        MP_CTX.Process(target=_throughput_sender, args=(barrier, signaling_url, size_mb, result_queue)),
        MP_CTX.Process(target=_throughput_receiver, args=(barrier, signaling_url, size_mb, result_queue)),
    ]
    for p in procs:
        p.start()
    results = [result_queue.get(timeout=120) for _ in procs]
    for p in procs:
        p.join(timeout=15)
    timings = next(r["timings"] for r in results if "timings" in r)
    mbps = [(size_mb * 8) / t for t in timings]  # Mbit/s
    return {
        "size_mb": size_mb,
        "mean_mbps": statistics.mean(mbps),
        "median_ms": statistics.median(timings) * 1000,
        "p90_ms": sorted(timings)[int(0.9 * len(timings)) - 1] * 1000 if len(timings) > 1 else timings[0] * 1000,
    }


def _inflight_sender(barrier, signaling_url: str, n: int, result_queue):
    import quic_dist
    import torch.distributed as dist

    quic_dist.init_process_group(
        signaling_url=signaling_url, rank=0, world_size=2, job_id=f"bench_if_{n}", timeout=timedelta(seconds=60)
    )
    tensors = [torch.randn(IN_FLIGHT_PAYLOAD_MB * MB // 4, dtype=torch.float32) for _ in range(n)]
    # Force connection establishment (hole-punch + QUIC handshake) BEFORE
    # the timed region - otherwise it's a one-time ~5s cost that swamps
    # the actual transfer time being measured here, especially at n=1.
    dist.send(torch.zeros(1), dst=1, tag=9999)
    barrier.wait()
    t0 = time.perf_counter()
    works = [dist.isend(tensors[i], dst=1, tag=i) for i in range(n)]
    for w in works:
        w.wait()
    elapsed = time.perf_counter() - t0
    dist.barrier()
    result_queue.put({"rank": 0, "elapsed": elapsed})

    pg = dist.distributed_c10d._get_default_group()
    dist.destroy_process_group()
    pg.close_connections()


def _inflight_receiver(barrier, signaling_url: str, n: int, result_queue):
    import quic_dist
    import torch.distributed as dist

    quic_dist.init_process_group(
        signaling_url=signaling_url, rank=1, world_size=2, job_id=f"bench_if_{n}", timeout=timedelta(seconds=60)
    )
    outs = [torch.zeros(IN_FLIGHT_PAYLOAD_MB * MB // 4, dtype=torch.float32) for _ in range(n)]
    dist.recv(torch.zeros(1), src=0, tag=9999)  # matching warmup - see sender
    barrier.wait()
    works = [dist.irecv(outs[i], src=0, tag=i) for i in range(n)]
    for w in works:
        w.wait()
    dist.barrier()
    result_queue.put({"rank": 1})

    pg = dist.distributed_c10d._get_default_group()
    dist.destroy_process_group()
    pg.close_connections()


def bench_inflight(signaling_url: str, n: int) -> dict:
    barrier = MP_CTX.Barrier(2)
    result_queue = MP_CTX.Queue()
    procs = [
        MP_CTX.Process(target=_inflight_sender, args=(barrier, signaling_url, n, result_queue)),
        MP_CTX.Process(target=_inflight_receiver, args=(barrier, signaling_url, n, result_queue)),
    ]
    for p in procs:
        p.start()
    results = [result_queue.get(timeout=120) for _ in procs]
    for p in procs:
        p.join(timeout=15)
    elapsed = next(r["elapsed"] for r in results if "elapsed" in r)
    total_mb = n * IN_FLIGHT_PAYLOAD_MB
    mbps = (total_mb * 8) / elapsed
    return {"n": n, "elapsed_s": elapsed, "aggregate_mbps": mbps}


def main() -> None:
    from _helpers import SignalingServer  # this package's own test infra, real subprocess

    server = SignalingServer()
    server.start()
    try:
        print("=== Throughput (single message, send/recv) ===")
        print(f"{'size (MB)':>10} {'mean Mbps':>12} {'median ms':>12} {'p90 ms':>10}")
        for size_mb in PAYLOAD_SIZES_MB:
            r = bench_throughput(server.url, size_mb)
            print(f"{r['size_mb']:>10} {r['mean_mbps']:>12.1f} {r['median_ms']:>12.2f} {r['p90_ms']:>10.2f}")

        print()
        print(f"=== In-flight isend/irecv sweep ({IN_FLIGHT_PAYLOAD_MB}MB each) ===")
        print(f"{'n in flight':>12} {'elapsed (s)':>12} {'aggregate Mbps':>16}")
        for n in IN_FLIGHT_COUNTS:
            r = bench_inflight(server.url, n)
            print(f"{r['n']:>12} {r['elapsed_s']:>12.3f} {r['aggregate_mbps']:>16.1f}")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
