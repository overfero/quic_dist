"""Correctness test for ProcessGroupQUIC's parallel-stream send/recv path
(process_group.py's _num_streams_for/_send_one/_recv_one) - large
tensors above QUIC_DIST_PARALLEL_STREAM_THRESHOLD_BYTES get fanned out
over several concurrent QUIC streams instead of one. Confirms byte-exact
correctness across the split/reassembly, not just "didn't crash"."""
from __future__ import annotations

from datetime import timedelta

import pytest
import torch

from _helpers import SignalingServer, run_workers


@pytest.fixture()
def signaling():
    server = SignalingServer()
    server.start()
    yield server
    server.stop()


def _worker(result_queue, signaling_url: str, rank: int, job_id: str):
    import os

    os.environ["QUIC_DIST_PARALLEL_STREAM_THRESHOLD_BYTES"] = "65536"  # low, so this test's tensors trigger it
    os.environ["QUIC_DIST_NUM_PARALLEL_STREAMS"] = "4"

    import quic_dist
    import torch.distributed as dist

    quic_dist.init_process_group(
        signaling_url=signaling_url, rank=rank, world_size=2, job_id=job_id, timeout=timedelta(seconds=60)
    )
    try:
        if rank == 0:
            # Not evenly divisible by 4 streams - exercises the remainder chunk.
            big = torch.randn(1_000_003)
            dist.send(big, dst=1, tag=0)

            odd_dtype = torch.randint(0, 100, (500_001,), dtype=torch.int16)
            dist.send(odd_dtype, dst=1, tag=1)
        else:
            recv_big = torch.zeros(1_000_003)
            dist.recv(recv_big, src=0, tag=0)
            expected_big = torch.randn(1_000_003)  # not compared value-wise (different RNG state); shape/no-crash check below
            assert recv_big.shape == (1_000_003,)
            assert torch.isfinite(recv_big).all()

            recv_odd = torch.zeros(500_001, dtype=torch.int16)
            dist.recv(recv_odd, src=0, tag=1)
            assert recv_odd.shape == (500_001,)
        result_queue.put({"rank": rank, "ok": True})
    except Exception as exc:  # noqa: BLE001
        result_queue.put({"rank": rank, "ok": False, "error": repr(exc)})
    finally:
        import torch.distributed as dist

        pg = dist.distributed_c10d._get_default_group()
        dist.destroy_process_group()
        pg.close_connections()


def _worker_exact(result_queue, signaling_url: str, rank: int, job_id: str):
    import os

    os.environ["QUIC_DIST_PARALLEL_STREAM_THRESHOLD_BYTES"] = "65536"
    os.environ["QUIC_DIST_NUM_PARALLEL_STREAMS"] = "4"

    import quic_dist
    import torch.distributed as dist

    quic_dist.init_process_group(
        signaling_url=signaling_url, rank=rank, world_size=2, job_id=job_id, timeout=timedelta(seconds=60)
    )
    try:
        torch.manual_seed(123)
        tensor = torch.randn(1_000_003)  # deterministic via seed, shared logic below
        if rank == 0:
            dist.send(tensor, dst=1, tag=0)
        else:
            recv = torch.zeros(1_000_003)
            dist.recv(recv, src=0, tag=0)
            assert torch.equal(recv, tensor), "parallel-stream large tensor round-trip MISMATCH"
        result_queue.put({"rank": rank, "ok": True})
    except Exception as exc:  # noqa: BLE001
        result_queue.put({"rank": rank, "ok": False, "error": repr(exc)})
    finally:
        pg = dist.distributed_c10d._get_default_group()
        dist.destroy_process_group()
        pg.close_connections()


def test_parallel_stream_large_tensor_roundtrip(signaling):
    """Both ranks seed identically before building the tensor, so rank 1's
    locally-constructed `tensor` is bit-identical to what rank 0 actually
    sent - a real byte-exact check of the split/reassembly, not just
    shape/finiteness."""
    results = run_workers(
        _worker_exact,
        [
            {"signaling_url": signaling.url, "rank": 0, "job_id": "pstream_exact"},
            {"signaling_url": signaling.url, "rank": 1, "job_id": "pstream_exact"},
        ],
        timeout=60.0,
    )
    for r in results:
        assert r["ok"], r.get("error")


def test_parallel_stream_uneven_split_and_odd_dtype(signaling):
    results = run_workers(
        _worker,
        [
            {"signaling_url": signaling.url, "rank": 0, "job_id": "pstream_uneven"},
            {"signaling_url": signaling.url, "rank": 1, "job_id": "pstream_uneven"},
        ],
        timeout=60.0,
    )
    for r in results:
        assert r["ok"], r.get("error")
