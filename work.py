"""Real async completion for `isend`/`irecv` - runs the underlying
blocking Rust call (`send_on_channel`/`recv_any`, both already release
the GIL internally via `py.detach()` - see `rust/src/quic_engine/`) on a
background thread, wrapped as a `torch.futures.Future` and converted to
a genuine `torch.distributed.Work` object via
`_create_work_from_future` - the exact mechanism torch's own reference
custom-backend implementation
(`torch/testing/_internal/distributed/multi_threaded_pg.py`'s `ret_work`
helper) uses.

This is what makes "the GPU should be able to compute while network
transfers are occurring" (the parallel-training prompt's Phase 6) real,
not aspirational: the blocking call genuinely runs on its own OS thread,
so the calling Python thread (and any CUDA work it launches) is never
blocked waiting for it - `work.wait()` is the only place that actually
blocks, and only if/when the caller chooses to call it.
"""
from __future__ import annotations

import concurrent.futures
from typing import Callable, TypeVar

import torch.futures
from torch._C._distributed_c10d import _create_work_from_future

_T = TypeVar("_T")

# One shared pool for the whole process - each in-flight isend/irecv
# occupies one worker thread for its duration (blocked inside the Rust
# call, not spinning), so this bounds real max-concurrent-in-flight-
# operations. Sized generously since these are I/O-bound waits, not
# CPU-bound work - matches this project's own established pattern
# (`asyncio.to_thread`'s default executor) for bridging a blocking
# native call into an async-shaped API without a dedicated thread per
# call.
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=64, thread_name_prefix="quic_dist-work")


def submit_as_work(fn: Callable[[], _T]) -> "torch.distributed.Work":
    """Runs `fn()` on a background thread; returns a real `Work` object
    whose `.wait()` blocks until `fn()` returns (or re-raises whatever it
    raised) and whose `.get_future()` gives a real `torch.futures.Future`
    for genuine async chaining."""
    fut: torch.futures.Future = torch.futures.Future()

    def _run() -> None:
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - must propagate through the Future, not swallow
            fut.set_exception(exc)
        else:
            fut.set_result(result)

    _EXECUTOR.submit(_run)
    return _create_work_from_future(fut)
