"""Auto-reconnect: ProcessGroupQUIC._get_or_connect detects a dead peer
connection (its dispatch thread observed the underlying QUIC connection
close) and transparently re-establishes it - a real hole-punch and QUIC
handshake, not a cached/fake one - on the next send/recv, rather than
leaving every future call to that peer permanently broken.

Simulates the death directly (sets the same `_closed_exc` state the real
dispatch thread sets when `recv_any()` errors) rather than waiting for a
real 45s idle_timeout - that keeps the test fast while still exercising
the actual recovery code path for real: what's being verified here is
that _get_or_connect notices and heals a dead connection, not how death
gets detected in the first place (already covered by the dispatch loop
itself, exercised implicitly by every other test in this suite)."""
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
        signaling_url=signaling_url, rank=rank, world_size=2, job_id=job_id, timeout=timedelta(seconds=60)
    )
    try:
        pg = dist.distributed_c10d._get_default_group()

        if rank == 0:
            dist.send(torch.tensor([1.0]), dst=1, tag=0)

            first_conn = pg._peers[1]
            first_conn.close()  # real close, so the OTHER side's dispatch thread also observes this connection ending
            first_conn._closed_exc = ConnectionError("simulated death for this test")

            dist.send(torch.tensor([2.0]), dst=1, tag=1)  # must trigger a real reconnect, not raise
            second_conn = pg._peers[1]
            assert second_conn is not first_conn, "reconnect should have replaced the dead connection object"
        else:
            recv1 = torch.zeros(1)
            dist.recv(recv1, src=0, tag=0)
            assert recv1.item() == 1.0

            recv2 = torch.zeros(1)
            dist.recv(recv2, src=0, tag=1)  # must succeed against the NEW connection rank0 reconnected with
            assert recv2.item() == 2.0

        dist.barrier()
        result_queue.put({"rank": rank, "ok": True})
    except Exception as exc:  # noqa: BLE001
        result_queue.put({"rank": rank, "ok": False, "error": repr(exc)})
    finally:
        pg = dist.distributed_c10d._get_default_group()
        dist.destroy_process_group()
        pg.close_connections()


def test_auto_reconnect_after_connection_death(signaling):
    results = run_workers(
        _worker,
        [
            {"signaling_url": signaling.url, "rank": 0, "job_id": "reconnect_test"},
            {"signaling_url": signaling.url, "rank": 1, "job_id": "reconnect_test"},
        ],
        timeout=60.0,
    )
    for r in results:
        assert r["ok"], r.get("error")
