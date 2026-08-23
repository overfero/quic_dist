"""Pipeline-parallel-shaped correctness test: several microbatches (each
its own `tag`) genuinely in flight concurrently through `ProcessGroupQUIC`
- forward activations one direction, backward gradients the other -
proving no cross-microbatch corruption under real concurrent multiplexed
traffic. Mirrors `tests/transport/test26_quic_broker_multiplexing.py`'s
"one shared connection backs every channel, none of them corrupt each
other" pattern, applied to send/recv/isend/irecv through the standard
torch.distributed API instead of the raw broker."""
from __future__ import annotations

from datetime import timedelta

import pytest
import torch

from _helpers import SignalingServer, run_workers

NUM_MICROBATCHES = 8
ACTIVATION_SHAPE = (4, 16)  # (microbatch, hidden) - small, real-shape-like


@pytest.fixture()
def signaling():
    server = SignalingServer()
    server.start()
    yield server
    server.stop()


def _teardown():
    import torch.distributed as dist

    pg = dist.distributed_c10d._get_default_group()
    dist.destroy_process_group()
    close = getattr(pg, "close_connections", None)
    if close is not None:
        close()


def _stage0_worker(result_queue, signaling_url: str, job_id: str):
    """Stage 0: sends forward activations for every microbatch (tags
    0..N-1), all issued back-to-back via isend (so they're genuinely
    concurrent, not serialized), then receives back gradients on the
    same tags and checks they match the expected backward-pass values."""
    import quic_dist
    import torch.distributed as dist

    quic_dist.init_process_group(
        signaling_url=signaling_url, rank=0, world_size=2, job_id=job_id, timeout=timedelta(seconds=60)
    )
    try:
        activations = [torch.randn(*ACTIVATION_SHAPE) + i for i in range(NUM_MICROBATCHES)]
        send_works = [dist.isend(activations[i], dst=1, tag=i) for i in range(NUM_MICROBATCHES)]
        for w in send_works:
            w.wait()

        grads = [torch.zeros(*ACTIVATION_SHAPE) for _ in range(NUM_MICROBATCHES)]
        recv_works = [dist.irecv(grads[i], src=1, tag=i) for i in range(NUM_MICROBATCHES)]
        for w in recv_works:
            w.wait()

        for i in range(NUM_MICROBATCHES):
            expected_grad = (activations[i] * 2.0) + 1.0  # stage1's fake "backward" transform
            assert torch.allclose(grads[i], expected_grad), f"microbatch {i}: gradient corrupted or cross-mixed"

        result_queue.put({"rank": 0, "ok": True})
    except Exception as exc:  # noqa: BLE001
        result_queue.put({"rank": 0, "ok": False, "error": repr(exc)})
    finally:
        _teardown()


def _stage1_worker(result_queue, signaling_url: str, job_id: str):
    """Stage 1: receives forward activations for every microbatch
    (in whatever order they happen to arrive - irecv issued for all tags
    up front), applies a deterministic fake "backward" transform, and
    sends the result back on the same tag."""
    import quic_dist
    import torch.distributed as dist

    quic_dist.init_process_group(
        signaling_url=signaling_url, rank=1, world_size=2, job_id=job_id, timeout=timedelta(seconds=60)
    )
    try:
        activations = [torch.zeros(*ACTIVATION_SHAPE) for _ in range(NUM_MICROBATCHES)]
        recv_works = [dist.irecv(activations[i], src=0, tag=i) for i in range(NUM_MICROBATCHES)]
        for w in recv_works:
            w.wait()

        grads = [(activations[i] * 2.0) + 1.0 for i in range(NUM_MICROBATCHES)]
        send_works = [dist.isend(grads[i], dst=0, tag=i) for i in range(NUM_MICROBATCHES)]
        for w in send_works:
            w.wait()

        result_queue.put({"rank": 1, "ok": True})
    except Exception as exc:  # noqa: BLE001
        result_queue.put({"rank": 1, "ok": False, "error": repr(exc)})
    finally:
        _teardown()


def test_pipeline_multiple_microbatches_no_cross_corruption(signaling):
    results = run_workers(
        _dual_stage_dispatch,
        [
            {"signaling_url": signaling.url, "job_id": "pipeline", "rank": 0},
            {"signaling_url": signaling.url, "job_id": "pipeline", "rank": 1},
        ],
        timeout=90.0,
    )
    for r in results:
        assert r["ok"], r.get("error")


def _dual_stage_dispatch(result_queue, signaling_url: str, job_id: str, rank: int):
    if rank == 0:
        _stage0_worker(result_queue, signaling_url, job_id)
    else:
        _stage1_worker(result_queue, signaling_url, job_id)
