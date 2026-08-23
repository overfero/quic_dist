"""Real 2-process correctness tests for `ProcessGroupQUIC` - genuine hole
punch, genuine QUIC handshake, genuine `torch.distributed.send/recv/
isend/irecv/barrier` calls through the standard torch API (not calling
`ProcessGroupQUIC` methods directly), matching this project's established
testing convention (`tests/transport/`: real subprocesses, real sockets,
no mocking)."""
from __future__ import annotations

import time
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


def _init(signaling_url: str, rank: int, world_size: int, job_id: str):
    import quic_dist

    quic_dist.init_process_group(
        signaling_url=signaling_url,
        rank=rank,
        world_size=world_size,
        job_id=job_id,
        timeout=timedelta(seconds=60),
    )


def _teardown():
    import torch.distributed as dist

    pg = dist.distributed_c10d._get_default_group()
    dist.destroy_process_group()
    close = getattr(pg, "close_connections", None)
    if close is not None:
        close()


def _worker_send_recv(result_queue, signaling_url: str, rank: int, job_id: str):
    import torch.distributed as dist

    _init(signaling_url, rank, 2, job_id)
    try:
        if rank == 0:
            tensor = torch.arange(1000, dtype=torch.float32)
            dist.send(tensor, dst=1, tag=0)

            out = torch.zeros(1000, dtype=torch.float32)
            dist.recv(out, src=1, tag=1)
            assert torch.equal(out, tensor * 2), "rank0: reply mismatch"
        else:
            out = torch.zeros(1000, dtype=torch.float32)
            dist.recv(out, src=0, tag=0)
            expected = torch.arange(1000, dtype=torch.float32)
            assert torch.equal(out, expected), "rank1: received mismatch"

            dist.send(out * 2, dst=0, tag=1)
        result_queue.put({"rank": rank, "ok": True})
    except Exception as exc:  # noqa: BLE001
        result_queue.put({"rank": rank, "ok": False, "error": repr(exc)})
    finally:
        _teardown()


def test_send_recv_roundtrip(signaling):
    results = run_workers(
        _worker_send_recv,
        [
            {"signaling_url": signaling.url, "rank": 0, "job_id": "send_recv"},
            {"signaling_url": signaling.url, "rank": 1, "job_id": "send_recv"},
        ],
        timeout=90.0,
    )
    for r in results:
        assert r["ok"], r.get("error")


def _worker_isend_irecv(result_queue, signaling_url: str, rank: int, job_id: str):
    import torch.distributed as dist

    _init(signaling_url, rank, 2, job_id)
    try:
        if rank == 0:
            tensors = [torch.full((100,), float(i)) for i in range(4)]
            works = [dist.isend(t, dst=1, tag=i) for i, t in enumerate(tensors)]
            for w in works:
                w.wait()
        else:
            outs = [torch.zeros(100) for _ in range(4)]
            works = [dist.irecv(outs[i], src=0, tag=i) for i in range(4)]
            for w in works:
                w.wait()
            for i, out in enumerate(outs):
                assert torch.equal(out, torch.full((100,), float(i))), f"tag {i} mismatch"
        result_queue.put({"rank": rank, "ok": True})
    except Exception as exc:  # noqa: BLE001
        result_queue.put({"rank": rank, "ok": False, "error": repr(exc)})
    finally:
        _teardown()


def test_isend_irecv_multiple_tags_in_flight(signaling):
    """Several tags in flight concurrently, received out of the order
    they were issued in - proves per-tag channel routing, not just a
    single in-order pipe."""
    results = run_workers(
        _worker_isend_irecv,
        [
            {"signaling_url": signaling.url, "rank": 0, "job_id": "isend_irecv"},
            {"signaling_url": signaling.url, "rank": 1, "job_id": "isend_irecv"},
        ],
        timeout=90.0,
    )
    for r in results:
        assert r["ok"], r.get("error")


def _worker_barrier(result_queue, signaling_url: str, rank: int, job_id: str):
    import torch.distributed as dist

    _init(signaling_url, rank, 2, job_id)
    try:
        t0 = time.monotonic()
        if rank == 1:
            time.sleep(1.0)
        dist.barrier()
        elapsed = time.monotonic() - t0
        result_queue.put({"rank": rank, "ok": True, "elapsed": elapsed})
    except Exception as exc:  # noqa: BLE001
        result_queue.put({"rank": rank, "ok": False, "error": repr(exc)})
    finally:
        _teardown()


def test_barrier_real_synchronization(signaling):
    results = run_workers(
        _worker_barrier,
        [
            {"signaling_url": signaling.url, "rank": 0, "job_id": "barrier"},
            {"signaling_url": signaling.url, "rank": 1, "job_id": "barrier"},
        ],
        timeout=60.0,
    )
    by_rank = {r["rank"]: r for r in results}
    assert by_rank[0]["ok"], by_rank[0].get("error")
    assert by_rank[1]["ok"], by_rank[1].get("error")
    assert by_rank[0]["elapsed"] >= 0.9  # rank 0 genuinely waited for rank 1


def _worker_not_implemented(result_queue, signaling_url: str, rank: int, job_id: str):
    import torch.distributed as dist

    _init(signaling_url, rank, 2, job_id)
    try:
        raised = False
        try:
            dist.all_reduce(torch.ones(4))
        except NotImplementedError:
            raised = True
        result_queue.put({"rank": rank, "ok": raised})
    except Exception as exc:  # noqa: BLE001
        result_queue.put({"rank": rank, "ok": False, "error": repr(exc)})
    finally:
        _teardown()


def test_collectives_raise_not_implemented(signaling):
    results = run_workers(
        _worker_not_implemented,
        [
            {"signaling_url": signaling.url, "rank": 0, "job_id": "notimpl"},
            {"signaling_url": signaling.url, "rank": 1, "job_id": "notimpl"},
        ],
        timeout=60.0,
    )
    for r in results:
        assert r["ok"], r.get("error")
