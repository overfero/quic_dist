"""`ProcessGroupQUIC`: a real `torch.distributed.ProcessGroup` backend,
registered as `"quic"` (`dist.Backend.register_backend`), so any PyTorch
code - Transformers, TRL, PEFT, a hand-written training loop - can use
`dist.init_process_group(backend="quic", ...)` without knowing QUIC is
involved at all. This is the general-purpose counterpart to this
project's existing `vllm/transport/pipeline_bootstrap.py`, which
deliberately bypasses `torch.distributed` entirely (a vLLM-internal
hack, not a reusable library) - see the accompanying plan for the full
rationale.

Scope, matching the parallel-training prompt's own explicit
prioritization ("do NOT start by implementing every collective
operation"): `send`/`recv`/`isend`/`irecv`/`barrier` only. Every other
`ProcessGroup` method (`allreduce`, `allgather`, `broadcast`, ...) raises
`NotImplementedError` with a clear message - never a silent no-op, never
a silent fallback to another backend (the prompt's own design
constraints #12/#13).

Connections to peers are established LAZILY, one real QUIC connection
(`_rust_quic_engine.PyMultiplexedConnectionDriver` - the same Rust-native
multi-channel driver that backs this project's own `--transport
quic-shared`, loaded directly from its compiled .so, not through the
vLLM package - see `_load_rust_quic_engine` below) per PEER, not per
message. Each distinct `tag` (torch's own `send`/`recv`/`isend`/`irecv`
parameter) maps to its
own named channel/stream within that connection - multiple microbatches
genuinely in flight, independently flow-controlled by quinn-proto itself
(no hand-rolled backpressure), no head-of-line blocking between tags.
Hole-punch reuses `peer.py` unmodified, exactly like every other backend
in this project.
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _store_based_barrier

from quic_dist.tensor import deserialize_tensor, serialize_tensor
from quic_dist.work import submit_as_work

def _load_rust_quic_engine():
    """The compiled PyO3 extension (`rust/src/quic_engine/python`, crate
    name `_rust_quic_engine`) is a standalone Rust crate with zero
    dependency on vLLM's own Python code. It currently happens to be
    *built into* vLLM's package directory only because that's where this
    session's `build_rust.sh` copies it for vLLM's own transport code to
    `import vllm._rust_quic_engine` - not because `quic_dist` (a general-
    purpose torch.distributed backend, unrelated to inference) conceptually
    needs vLLM installed. Loaded directly by file path instead, so
    `quic_dist` never requires `import vllm` to succeed. Checks known
    build-output locations only (no recursive filesystem search - the
    Rust build directory alone is several GB with tens of thousands of
    files, far too slow to scan on every import)."""
    import importlib.util

    repo_root = Path(__file__).resolve().parent.parent
    candidates = []
    if os.environ.get("QUIC_DIST_RUST_ENGINE_SO"):
        candidates.append(Path(os.environ["QUIC_DIST_RUST_ENGINE_SO"]))
    candidates += list((repo_root / "vllm" / "vllm").glob("_rust_quic_engine*.so"))
    candidates += list((repo_root / "quic_dist").glob("_rust_quic_engine*.so"))
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location("_rust_quic_engine", candidate)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    raise ImportError(
        "Could not locate the compiled _rust_quic_engine extension (build it "
        "from rust/src/quic_engine/python via ./build_rust.sh). Set "
        "QUIC_DIST_RUST_ENGINE_SO to its .so path if it isn't at the usual "
        f"vllm/vllm/_rust_quic_engine*.so location under {repo_root}."
    )


_qe = _load_rust_quic_engine()


def _locate_existing_transport_dir() -> Path:
    """`peer.py` (hole-punch/STUN, unmodified, reused as-is) currently
    lives under vLLM's own tree (`vllm/udp_holepunch/`) - genuinely
    reusable, generic code, just not yet relocated out of vLLM's
    directory. Checked as a known relative path (not a recursive search -
    same reasoning as `_load_rust_quic_engine` above)."""
    repo_root = Path(__file__).resolve().parent.parent
    candidates = []
    if os.environ.get("VLLM_UDP_TRANSPORT_DIR"):
        candidates.append(Path(os.environ["VLLM_UDP_TRANSPORT_DIR"]))
    candidates.append(repo_root / "vllm" / "udp_holepunch")
    candidates.append(repo_root / "udp_holepunch")
    for candidate in candidates:
        if (candidate / "peer.py").exists():
            return candidate
    raise RuntimeError(
        "Could not locate the existing UDP hole-punch transport (peer.py). "
        "Set VLLM_UDP_TRANSPORT_DIR to the directory that contains it."
    )


_existing_transport_dir = _locate_existing_transport_dir()
if str(_existing_transport_dir) not in sys.path:
    sys.path.insert(0, str(_existing_transport_dir))

import peer as _hp  # noqa: E402  (unmodified - hole-punch/STUN only)

_SIGNALING_URL_ENV = "QUIC_DIST_SIGNALING_URL"
_FALLBACK_WINDOW_BYTES = 1024 * 1024
_DEFAULT_IDLE_TIMEOUT_S = 45.0
_DEFAULT_MAX_MESSAGE_BYTES = 2 * 1024 * 1024 * 1024
_NO_TIMEOUT_MS = 2**31 - 1
_DRAIN_TIMEOUT_MS = 3000


class _PeerConnection:
    """One real QUIC connection to one peer rank, with per-tag inbound
    queues fanned out by a background dispatch thread - the synchronous
    (threading.Thread/queue.Queue, no asyncio loop needed here) analogue
    of `quic_broker.py`'s own `_dispatch_loop`/`_channel_queues`
    mechanism, applied directly instead of through a local Unix-socket
    IPC hop (this class IS the consumer - no separate client process)."""

    def __init__(self, driver: "_qe.PyMultiplexedConnectionDriver") -> None:
        self._driver = driver
        self._queues: dict[str, "queue.Queue"] = {}
        self._queues_lock = threading.Lock()
        self._closed_exc: Exception | None = None
        self._dispatch_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
        self._dispatch_thread.start()

    def _queue_for(self, tag: str) -> "queue.Queue":
        with self._queues_lock:
            q = self._queues.get(tag)
            if q is None:
                q = self._queues[tag] = queue.Queue()
                if self._closed_exc is not None:
                    q.put(self._closed_exc)
            return q

    def _dispatch_loop(self) -> None:
        while True:
            try:
                tag, data = self._driver.recv_any(_NO_TIMEOUT_MS)
            except ValueError as exc:
                self._closed_exc = ConnectionError(f"ProcessGroupQUIC peer connection closed: {exc}")
                with self._queues_lock:
                    for q in self._queues.values():
                        q.put(self._closed_exc)
                return
            self._queue_for(tag).put(bytes(data))

    def send(self, tag: str, payload: bytes, timeout_s: float) -> None:
        self._driver.send_on_channel(tag, payload, int(timeout_s * 1000))

    def recv(self, tag: str, timeout_s: float | None) -> bytes:
        q = self._queue_for(tag)
        try:
            item = q.get(timeout=timeout_s)
        except queue.Empty:
            raise TimeoutError(f"ProcessGroupQUIC recv(tag={tag!r}) timed out after {timeout_s}s") from None
        if isinstance(item, Exception):
            q.put(item)  # leave it for any other waiter on this tag
            raise item
        return item

    def close(self) -> None:
        self._driver.close(_DRAIN_TIMEOUT_MS)


class ProcessGroupQUIC(dist.ProcessGroup):
    def __init__(
        self,
        store: dist.Store,
        rank: int,
        world_size: int,
        timeout: timedelta,
        *,
        signaling_url: str,
        job_id: str = "default",
    ) -> None:
        super().__init__(rank, world_size)
        self._store = store
        self._rank = rank
        self._world_size = world_size
        self._timeout = timeout
        self._signaling_url = signaling_url
        self._job_id = job_id
        self._peers: dict[int, _PeerConnection] = {}
        self._peers_lock = threading.Lock()
        self._next_message_id = 0
        self._message_id_lock = threading.Lock()

    # ---- connection management ----

    def _self_id(self, rank: int) -> str:
        return f"quic_dist:{self._job_id}:rank{rank}"

    def _get_or_connect(self, peer_rank: int) -> _PeerConnection:
        with self._peers_lock:
            conn = self._peers.get(peer_rank)
            if conn is not None:
                return conn
        conn = self._connect_to_peer(peer_rank)
        with self._peers_lock:
            self._peers[peer_rank] = conn
        return conn

    def _connect_to_peer(self, peer_rank: int) -> _PeerConnection:
        """Real hole-punch (unmodified `peer.py`) then a real QUIC
        handshake via `PyMultiplexedConnectionDriver` - same pattern as
        `quic_broker.py`'s `_connect_async`, just synchronous (a short-
        lived asyncio loop for the hole-punch phase only - unlike
        `quic_broker.py`, nothing here needs a persistent loop
        afterward, since there's no local Unix-socket IPC hop: this
        class talks to the Rust driver directly)."""
        import asyncio
        import contextlib
        import socket

        self_id = self._self_id(self._rank)
        peer_id = self._self_id(peer_rank)
        # Deterministic role assignment - lower rank listens (QUIC
        # server), matching pipeline_bootstrap.py's own convention.
        is_client = self._rank > peer_rank

        async def _connect_async() -> "_qe.PyMultiplexedConnectionDriver":
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            for opt in (socket.SO_RCVBUF, socket.SO_SNDBUF):
                with contextlib.suppress(OSError):
                    sock.setsockopt(socket.SOL_SOCKET, opt, _hp.SOCKET_BUFFER_REQUEST)
            sock.bind(("0.0.0.0", 0))

            own_ip, own_port = None, sock.getsockname()[1]
            reg_resp = _hp.register(self._signaling_url, self_id, own_port)
            own_ip = own_ip or reg_resp["public_ip"]

            peer_info = _hp.wait_for_peer(self._signaling_url, self_id, peer_id)
            coordinator_peer_addr = (peer_info["public_ip"], peer_info["udp_port"])
            delay = peer_info["start_at"] - time.time()
            if delay > 0:
                await asyncio.sleep(delay)

            loop = asyncio.get_running_loop()
            transport, protocol = await loop.create_datagram_endpoint(
                lambda: _hp.PeerProtocol(coordinator_peer_addr), sock=sock
            )
            punch_task = asyncio.create_task(_hp.punch_loop(protocol))
            deadline = time.monotonic() + self._timeout.total_seconds()
            while not protocol.established and time.monotonic() < deadline:
                await asyncio.sleep(0.1)
            punch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await punch_task
            if not protocol.established:
                transport.close()
                raise ConnectionError(
                    f"ProcessGroupQUIC: hole punch to rank {peer_rank} failed within "
                    f"{self._timeout.total_seconds()}s"
                )

            # Window sizing: stream_receive_window is 6x the base `window`
            # here, NOT tied 1:1 like vllm/transport/quic_transport.py's
            # identical derivation. Empirically tuned via a real
            # cross-machine benchmark (local <-> akun7, 65ms RTT, 250+
            # Mbps raw UDP capacity on the same path per peer.py's own
            # benchmark): 1x measured ~39 Mbps (window*8/RTT-limited,
            # matches theory); raising to 8x reintroduced the exact stall
            # quic_transport.py's 1x cap was originally, deliberately
            # built to prevent (self-induced loss from the congestion
            # controller bursting past the real OS buffer - that fix was
            # validated on loopback, where a small window barely costs
            # anything since it recycles almost instantly at ~0 RTT); 6x
            # measured ~100 Mbps with no stall, confirmed twice. Binary
            # search stopped at 6x (safe margin below the confirmed 8x
            # failure) rather than searching for the exact boundary - a
            # pipeline-parallel training link is exactly the genuine
            # multi-hop, real-RTT case this cap was never tuned for.
            # Deliberately changed only here, not in the shared vLLM
            # transports (out of scope, and their tuning is validated
            # against a different, live deployment).
            try:
                granted_rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                granted_sndbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
                window = max(64 * 1024, min(granted_rcvbuf, granted_sndbuf))
            except OSError:
                window = _FALLBACK_WINDOW_BYTES
            stream_window = 6 * window
            idle_timeout_ms = max(1, int(_DEFAULT_IDLE_TIMEOUT_S * 1000))
            handshake_timeout_ms = max(1, int(self._timeout.total_seconds() * 1000))

            try:
                if is_client:
                    driver = _qe.PyMultiplexedConnectionDriver.connect_client(
                        sock.fileno(), coordinator_peer_addr[0], coordinator_peer_addr[1],
                        "quic-dist-v1", idle_timeout_ms, 8 * window, 8 * window, stream_window,
                        8 * window, _DEFAULT_MAX_MESSAGE_BYTES, handshake_timeout_ms,
                    )
                else:
                    driver = _qe.PyMultiplexedConnectionDriver.connect_server(
                        sock.fileno(), idle_timeout_ms, 8 * window, 8 * window, stream_window,
                        8 * window, _DEFAULT_MAX_MESSAGE_BYTES, handshake_timeout_ms,
                    )
            except ValueError as exc:
                transport.close()
                raise ConnectionError(
                    f"ProcessGroupQUIC: QUIC handshake with rank {peer_rank} failed: {exc}"
                ) from None
            transport.close()  # asyncio's job (hole-punch) is done - Rust owns the socket now
            return driver

        driver = asyncio.new_event_loop().run_until_complete(_connect_async())
        return _PeerConnection(driver)

    def _next_msg_id(self) -> int:
        with self._message_id_lock:
            self._next_message_id += 1
            return self._next_message_id

    # ---- point-to-point ----

    def _send_one(self, tensor: torch.Tensor, dst: int, tag: int, msg_id: int) -> None:
        payload = serialize_tensor(tensor, message_id=msg_id, microbatch_id=tag, tensor_id=0)
        conn = self._get_or_connect(dst)
        conn.send(str(tag), payload, self._timeout.total_seconds())

    def _recv_one(self, tensor: torch.Tensor, src: int, tag: int) -> None:
        conn = self._get_or_connect(src)
        payload = conn.recv(str(tag), self._timeout.total_seconds())
        received, meta = deserialize_tensor(payload)
        if tuple(received.shape) != tuple(tensor.shape) or received.dtype != tensor.dtype:
            raise RuntimeError(
                f"ProcessGroupQUIC.recv: shape/dtype mismatch - expected "
                f"{tuple(tensor.shape)}/{tensor.dtype}, got {tuple(received.shape)}/"
                f"{received.dtype} (message_id={meta.message_id}, tensor_id={meta.tensor_id})"
            )
        tensor.copy_(received.to(tensor.device))

    def send(self, tensors: list[torch.Tensor], dst: int, tag: int = 0) -> "dist.Work":
        msg_id = self._next_msg_id()
        for tensor in tensors:
            self._send_one(tensor, dst, tag, msg_id)
        return submit_as_work(lambda: None)  # already done - matches send()'s "blocks until handed off" contract

    def recv(self, tensors: list[torch.Tensor], src: int, tag: int = 0) -> "dist.Work":
        for tensor in tensors:
            self._recv_one(tensor, src, tag)
        return submit_as_work(lambda: None)

    def isend(self, tensors: list[torch.Tensor], dst: int, tag: int = 0) -> "dist.Work":
        msg_id = self._next_msg_id()

        def _do() -> None:
            for tensor in tensors:
                self._send_one(tensor, dst, tag, msg_id)

        return submit_as_work(_do)

    def irecv(self, tensors: list[torch.Tensor], src: int, tag: int = 0) -> "dist.Work":
        def _do() -> None:
            for tensor in tensors:
                self._recv_one(tensor, src, tag)

        return submit_as_work(_do)

    # ---- barrier ----

    def barrier(self, opts=None) -> "dist.Work":
        _store_based_barrier(
            self._rank, self._store, f"quic_dist:{self._job_id}", self._world_size, self._timeout
        )
        return submit_as_work(lambda: None)

    # ---- explicitly not implemented (see module docstring) ----

    def _not_implemented(self, name: str):
        raise NotImplementedError(
            f"ProcessGroupQUIC.{name}() is not implemented - this backend currently only "
            "supports point-to-point communication (send/recv/isend/irecv/barrier), matching "
            "the pipeline-parallel-first scope of the parallel-training prompt this was built "
            "from. Collectives were deliberately deferred, not silently faked."
        )

    def allreduce(self, *a, **kw):
        self._not_implemented("allreduce")

    def allgather(self, *a, **kw):
        self._not_implemented("allgather")

    def broadcast(self, *a, **kw):
        self._not_implemented("broadcast")

    def reduce_scatter(self, *a, **kw):
        self._not_implemented("reduce_scatter")

    def scatter(self, *a, **kw):
        self._not_implemented("scatter")

    def gather(self, *a, **kw):
        self._not_implemented("gather")

    def alltoall(self, *a, **kw):
        self._not_implemented("alltoall")

    # ---- misc required overrides ----

    def size(self) -> int:
        return self._world_size

    def getBackendName(self) -> str:
        return "quic"

    def close_connections(self) -> None:
        """Not a torch.distributed.ProcessGroup API method - explicit
        cleanup for this backend's own peer connections. Call before
        process exit (torch itself never calls this)."""
        with self._peers_lock:
            peers = list(self._peers.values())
            self._peers.clear()
        for conn in peers:
            conn.close()


def _create_quic_pg(prefix_store: dist.Store, rank: int, world_size: int, timeout: timedelta) -> ProcessGroupQUIC:
    signaling_url = os.environ.get(_SIGNALING_URL_ENV)
    if not signaling_url:
        raise RuntimeError(
            f"ProcessGroupQUIC: {_SIGNALING_URL_ENV} environment variable is not set - "
            "either call quic_dist.init_process_group(...) (sets it for you) or set it "
            "yourself before calling dist.init_process_group(backend='quic', ...) directly"
        )
    job_id = os.environ.get("QUIC_DIST_JOB_ID", "default")
    return ProcessGroupQUIC(prefix_store, rank, world_size, timeout, signaling_url=signaling_url, job_id=job_id)


dist.Backend.register_backend("quic", _create_quic_pg, devices=["cpu", "cuda"])
