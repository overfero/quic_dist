"""`quic_dist`: a real `torch.distributed.ProcessGroup` backend over QUIC,
for pipeline-parallel distributed training (LoRA/QLoRA and similar)
between NAT'd machines - no cloud VPC, no port forwarding, no direct
reachability required, only that every rank can reach one shared HTTP
signaling server.

Minimal usage::

    import quic_dist

    quic_dist.init_process_group(
        signaling_url="https://my-signaling-server.example",
        rank=rank,
        world_size=world_size,
    )
    # from here on, this is a completely ordinary torch.distributed
    # process group - the standard API works as-is:
    import torch.distributed as dist
    dist.send(tensor, dst=1, tag=7)
    dist.recv(tensor, src=1, tag=7)
    dist.barrier()

Scope: point-to-point (`send`/`recv`/`isend`/`irecv`) and `barrier()`
only - see `quic_dist.process_group` for why collectives
(`all_reduce`/`all_gather`/etc.) deliberately raise `NotImplementedError`
rather than silently no-op or fall back to another backend.
"""
from __future__ import annotations

from datetime import timedelta

import torch.distributed as dist

# Side effect: registers the "quic" backend with
# dist.Backend.register_backend(...) - required before
# dist.init_process_group(backend="quic", ...) can succeed.
import quic_dist.process_group as process_group
from quic_dist.process_group import ProcessGroupQUIC
from quic_dist.store import QuicRendezvousStore
from quic_dist.tensor import TensorMetadata, deserialize_tensor, serialize_tensor
from quic_dist.work import submit_as_work

_KEPT_ALIVE_STORES: list[QuicRendezvousStore] = []

__all__ = [
    "ProcessGroupQUIC",
    "QuicRendezvousStore",
    "TensorMetadata",
    "deserialize_tensor",
    "serialize_tensor",
    "submit_as_work",
    "init_process_group",
]


def init_process_group(
    *,
    signaling_url: str,
    rank: int,
    world_size: int,
    job_id: str = "default",
    timeout: timedelta = timedelta(seconds=300),
) -> None:
    """Convenience wrapper around `dist.init_process_group(backend="quic",
    ...)`: builds the `QuicRendezvousStore` (backed by `signaling_url`)
    and passes `QUIC_DIST_SIGNALING_URL`/`QUIC_DIST_JOB_ID` through to the
    backend's registered creator function via the environment (torch's
    `Backend.register_backend` creator only receives
    `(prefix_store, rank, world_size, timeout)` - no channel for extra
    config - see `process_group._create_quic_pg`).

    `job_id` namespaces hole-punch peer IDs and the barrier's store key
    prefix, so multiple unrelated `quic_dist` jobs can share one
    signaling server without colliding on rank numbers.

    For direct control (e.g. a custom `Store`), call
    `dist.init_process_group(backend="quic", store=..., rank=..., \
world_size=..., timeout=...)` yourself instead - just set
    `QUIC_DIST_SIGNALING_URL` (and optionally `QUIC_DIST_JOB_ID`) first.
    """
    import os

    os.environ["QUIC_DIST_SIGNALING_URL"] = signaling_url
    os.environ["QUIC_DIST_JOB_ID"] = job_id

    store = QuicRendezvousStore(signaling_url, timeout=timeout)
    # torch wraps `store` in a C++-side PrefixStore before ProcessGroupQUIC
    # ever sees it; that wrapper does NOT keep the underlying Python object
    # alive on its own (confirmed by a real crash: dist.barrier() -> "Tried
    # to call pure virtual function Store::add" on the *second* use, after
    # this function's local `store` went out of scope and got garbage
    # collected). Keep an explicit strong reference for the process
    # lifetime - same pattern torch's own reference custom-backend tests
    # use for exactly this reason.
    _KEPT_ALIVE_STORES.append(store)
    dist.init_process_group(
        backend="quic",
        store=store,
        rank=rank,
        world_size=world_size,
        timeout=timeout,
    )
