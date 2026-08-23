"""ProcessGroupQUIC at world_size=3 - real, not simulated at world_size=2.
Every other test in this suite only ever exercises world_size=2; this is
the first real validation that a middle-of-pipeline rank can maintain
TWO separate peer connections concurrently (one to its predecessor, one
to its successor) and that the store-based barrier genuinely
synchronizes 3 processes, not just 2.

Topology: rank 0 -> rank 1 -> rank 2 (a real 3-stage pipeline shape).
rank 1 receives from rank 0, transforms, sends to rank 2 - the same rank
using two independently-established `_PeerConnection`s at once."""
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
    import quic_dist
    import torch.distributed as dist

    quic_dist.init_process_group(
        signaling_url=signaling_url, rank=rank, world_size=3, job_id=job_id, timeout=timedelta(seconds=60)
    )
    try:
        if rank == 0:
            tensor = torch.arange(100, dtype=torch.float32)
            dist.send(tensor, dst=1, tag=0)
        elif rank == 1:
            recv_buf = torch.zeros(100, dtype=torch.float32)
            dist.recv(recv_buf, src=0, tag=0)
            expected = torch.arange(100, dtype=torch.float32)
            assert torch.equal(recv_buf, expected), "rank1: received from rank0 mismatch"

            pg = dist.distributed_c10d._get_default_group()
            assert 0 in pg._peers and 1 not in pg._peers, "rank1 should have a connection to rank0 only so far"

            dist.send(recv_buf * 2, dst=2, tag=0)
            assert len(pg._peers) == 2, "rank1 should now hold TWO concurrent peer connections (rank0 and rank2)"
        else:  # rank == 2
            recv_buf = torch.zeros(100, dtype=torch.float32)
            dist.recv(recv_buf, src=1, tag=0)
            expected = torch.arange(100, dtype=torch.float32) * 2
            assert torch.equal(recv_buf, expected), "rank2: received from rank1 mismatch"

        # A real 3-way barrier - not just 2. Every rank must reach this
        # line before any of them proceeds past it.
        dist.barrier()
        result_queue.put({"rank": rank, "ok": True})
    except Exception as exc:  # noqa: BLE001
        result_queue.put({"rank": rank, "ok": False, "error": repr(exc)})
    finally:
        pg = dist.distributed_c10d._get_default_group()
        dist.destroy_process_group()
        pg.close_connections()


def test_three_rank_pipeline(signaling):
    results = run_workers(
        _worker,
        [
            {"signaling_url": signaling.url, "rank": 0, "job_id": "n3pipeline"},
            {"signaling_url": signaling.url, "rank": 1, "job_id": "n3pipeline"},
            {"signaling_url": signaling.url, "rank": 2, "job_id": "n3pipeline"},
        ],
        timeout=90.0,
    )
    for r in results:
        assert r["ok"], r.get("error")
