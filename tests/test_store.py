"""Correctness tests for `QuicRendezvousStore` against a real (unmodified)
signaling server subprocess - no mocking of the HTTP layer, matching this
project's established testing convention (real subprocesses, real
sockets, real network calls throughout `tests/transport/`)."""
from __future__ import annotations

import threading
import time
from datetime import timedelta

import pytest

from _helpers import SignalingServer, run_workers

from quic_dist.store import QuicRendezvousStore


@pytest.fixture()
def signaling():
    server = SignalingServer()
    server.start()
    yield server
    server.stop()


def test_set_get_roundtrip(signaling):
    store = QuicRendezvousStore(signaling.url, timeout=timedelta(seconds=5))
    store.set("k1", b"hello world")
    assert store.get("k1") == b"hello world"


def test_set_get_roundtrip_binary_bytes(signaling):
    store = QuicRendezvousStore(signaling.url, timeout=timedelta(seconds=5))
    payload = bytes(range(256))
    store.set("binary", payload)
    assert store.get("binary") == payload


def test_get_blocks_until_set(signaling):
    store = QuicRendezvousStore(signaling.url, timeout=timedelta(seconds=10))
    result = {}

    def _setter():
        time.sleep(1.0)
        store.set("delayed", b"arrived")

    t = threading.Thread(target=_setter)
    t.start()
    t0 = time.monotonic()
    result["value"] = store.get("delayed")
    elapsed = time.monotonic() - t0
    t.join()

    assert result["value"] == b"arrived"
    assert elapsed >= 0.9  # actually waited, didn't return stale/empty immediately


def test_get_times_out(signaling):
    store = QuicRendezvousStore(signaling.url, timeout=timedelta(seconds=1))
    with pytest.raises(RuntimeError):
        store.get("never-set")


def test_wait_success_and_timeout(signaling):
    store = QuicRendezvousStore(signaling.url, timeout=timedelta(seconds=5))
    store.set("present", b"x")
    store.wait(["present"])  # should return immediately, no raise

    fast_store = QuicRendezvousStore(signaling.url, timeout=timedelta(seconds=1))
    with pytest.raises(RuntimeError):
        fast_store.wait(["absent-key"])


def test_add_is_atomic_counter(signaling):
    store = QuicRendezvousStore(signaling.url, timeout=timedelta(seconds=5))
    assert store.add("counter", 5) == 5
    assert store.add("counter", 3) == 8
    assert store.add("counter", -2) == 6


def test_compare_set(signaling):
    store = QuicRendezvousStore(signaling.url, timeout=timedelta(seconds=5))
    # key absent: expected="" matches "not present"
    result = store.compare_set("cas", b"", b"first")
    assert result == b"first"
    assert store.get("cas") == b"first"

    # wrong expected: no-op, returns current value
    result = store.compare_set("cas", b"wrong", b"second")
    assert result == b"first"

    # right expected: swaps
    result = store.compare_set("cas", b"first", b"second")
    assert result == b"second"
    assert store.get("cas") == b"second"


def test_check_and_delete_key(signaling):
    store = QuicRendezvousStore(signaling.url, timeout=timedelta(seconds=5))
    assert store.check(["missing"]) is False
    store.set("present2", b"y")
    assert store.check(["present2"]) is True

    store.delete_key("present2")
    assert store.check(["present2"]) is False


def _barrier_worker(result_queue, signaling_url: str, rank: int, world_size: int):
    import torch.distributed.distributed_c10d as c10d

    from quic_dist.store import QuicRendezvousStore

    store = QuicRendezvousStore(signaling_url, timeout=timedelta(seconds=30))
    t0 = time.monotonic()
    if rank == 1:
        time.sleep(1.0)  # rank 0 must genuinely wait for rank 1
    c10d._store_based_barrier(rank, store, "test_barrier", world_size, timeout=timedelta(seconds=30))
    elapsed = time.monotonic() - t0
    result_queue.put({"rank": rank, "elapsed": elapsed})


def test_store_based_barrier_real_two_process(signaling):
    """Reuses torch's own internal barrier helper directly - proves
    QuicRendezvousStore satisfies its real contract (set/get/add/wait),
    not just each primitive in isolation."""
    results = run_workers(
        _barrier_worker,
        [
            {"signaling_url": signaling.url, "rank": 0, "world_size": 2},
            {"signaling_url": signaling.url, "rank": 1, "world_size": 2},
        ],
        timeout=60.0,
    )
    assert len(results) == 2
    by_rank = {r["rank"]: r for r in results}
    # rank 0 must have actually waited for rank 1's delayed arrival, not
    # returned immediately - this is the real point of the test.
    assert by_rank[0]["elapsed"] >= 0.9
