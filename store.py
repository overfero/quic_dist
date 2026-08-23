"""`QuicRendezvousStore`: a `torch.distributed.Store` implementation
backed by the existing UDP hole-punch signaling server's generic
key-value HTTP API (`udp_holepunch/signaling_server.py`'s `/kv/*`
endpoints - added specifically for this, see that file's module
docstring).

Why this exists: `dist.init_process_group()` needs a `Store` for its
initial rendezvous handshake (key exchange BEFORE any `ProcessGroup`
exists) and its own internal store-based barrier
(`torch.distributed.distributed_c10d._store_based_barrier`, confirmed by
reading its source directly - it calls exactly `set`/`get`/`add`/`wait`,
nothing else). The standard answer is `TCPStore`, but that requires rank
0 to be directly reachable by every other rank - exactly the NAT problem
this whole project's hole-punch machinery exists to avoid. This Store
routes the same `set`/`get`/`add`/`wait`/`compare_set` operations through
plain HTTP requests to the signaling server instead - no direct
reachability between ranks needed at all, only that every rank can reach
the (already centrally-reachable, e.g. via a zrok tunnel) signaling
server, same requirement every other backend in this project already has.

Confirmed directly in this environment (see the accompanying plan) that
`torch.distributed.Store` is subclassable from pure Python (pybind11
trampoline) but every method used here is a PURE VIRTUAL C++ method with
no usable default - `set`/`get`/`add`/`wait`/`compare_set` all raise
"Tried to call pure virtual function" if left un-overridden, so every one
of them needs a real implementation here, not just the ones a first
guess might assume are load-bearing.

`wait()` is implemented via client-side polling against `GET /kv/get?key=`
(key as a query param, not a path segment - `PrefixStore` always produces
keys containing "/", which breaks path-segment routing), matching
`peer.py`'s own `wait_for_peer` polling convention - no server-side
blocking/websocket machinery needed.
"""
from __future__ import annotations

import base64
import time
from datetime import timedelta

import requests
import torch.distributed as dist

_DEFAULT_POLL_INTERVAL_S = 0.2


def _b64encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _b64decode(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


class QuicRendezvousStore(dist.Store):
    """`signaling_url` is the SAME HTTP endpoint `peer.py`'s hole-punch
    calls already use (`_hp.register`/`_hp.wait_for_peer`) - one server,
    two purposes (hole-punch coordination + this generic KV store),
    genuinely unrelated request paths (`/register`,`/peer/*` vs `/kv/*`).
    """

    def __init__(self, signaling_url: str, timeout: timedelta = timedelta(seconds=300)) -> None:
        super().__init__()
        self._signaling_url = signaling_url.rstrip("/")
        self._timeout = timeout

    # ---- the real primitives - everything else in torch's Store API
    # that matters for rendezvous/barrier is expressed in terms of these
    # four, matching `_store_based_barrier`'s own usage exactly. ----

    def _request_with_retry(self, request_fn, deadline: float):
        """Runs `request_fn()` (a zero-arg call performing ONE HTTP
        request), retrying on transient network failures (timeout,
        connection reset/refused) until `deadline` (a
        `time.monotonic()` value). A real, observed failure mode over a
        zrok tunnel under concurrent multi-rank load - `add()` used to
        make exactly one `requests.post(..., timeout=10)` call with NO
        retry at all, and a real `dist.barrier()` call (this store's
        first real exerciser of `_store_based_barrier`'s `add()` usage)
        hit a bare `ReadTimeoutError` from that single call and crashed
        the whole rank, even though every other network hiccup in this
        class was already survivable via its own polling loop. HTTP
        status errors (4xx/5xx, via `raise_for_status()`) are NOT
        retried here - only request-level failures that mean no
        response was received at all."""
        while True:
            try:
                return request_fn()
            except requests.exceptions.RequestException:
                if time.monotonic() > deadline:
                    raise
                time.sleep(_DEFAULT_POLL_INTERVAL_S)

    def set(self, key: str, value: bytes | str) -> None:
        # torch's own internals (e.g. _store_based_barrier's
        # `store.set(last_worker_key, "1")`) call this with a plain str,
        # not bytes - confirmed by a real crash here. A real TCPStore
        # accepts both (the pybind11 binding UTF-8-encodes a str
        # automatically); this pure-Python override needs to do the same
        # coercion itself, there's no default to inherit it from.
        if isinstance(value, str):
            value = value.encode("utf-8")
        deadline = time.monotonic() + self._timeout.total_seconds()
        resp = self._request_with_retry(
            lambda: requests.post(
                f"{self._signaling_url}/kv/set",
                json={"key": key, "value": _b64encode(value)},
                timeout=10,
            ),
            deadline,
        )
        resp.raise_for_status()

    def get(self, key: str) -> bytes:
        # Torch's own Store.get() contract is "block until present" (it's
        # `wait([key])` + `get()` combined in the C++ base class - the
        # pybind trampoline for a pure-Python override does NOT get that
        # behavior for free, since we're replacing the whole method, not
        # extending it), so this polls exactly like `wait()` does below,
        # then returns the value - matches what callers of a real
        # `TCPStore`/`HashStore` actually observe.
        deadline = time.monotonic() + self._timeout.total_seconds()
        while True:
            resp = self._request_with_retry(
                lambda: requests.get(f"{self._signaling_url}/kv/get", params={"key": key}, timeout=10),
                deadline,
            )
            if resp.status_code == 200:
                return _b64decode(resp.json()["value"])
            if resp.status_code != 404:
                resp.raise_for_status()
            if time.monotonic() > deadline:
                raise RuntimeError(f"QuicRendezvousStore.get({key!r}) timed out after {self._timeout}")
            time.sleep(_DEFAULT_POLL_INTERVAL_S)

    def add(self, key: str, value: int) -> int:
        deadline = time.monotonic() + self._timeout.total_seconds()
        resp = self._request_with_retry(
            lambda: requests.post(
                f"{self._signaling_url}/kv/add",
                json={"key": key, "amount": value},
                timeout=10,
            ),
            deadline,
        )
        resp.raise_for_status()
        return int(resp.json()["value"])

    def compare_set(self, key: str, expected: bytes | str, desired: bytes | str) -> bytes:
        if isinstance(expected, str):
            expected = expected.encode("utf-8")
        if isinstance(desired, str):
            desired = desired.encode("utf-8")
        deadline = time.monotonic() + self._timeout.total_seconds()
        resp = self._request_with_retry(
            lambda: requests.post(
                f"{self._signaling_url}/kv/compare_set",
                json={"key": key, "expected": _b64encode(expected), "desired": _b64encode(desired)},
                timeout=10,
            ),
            deadline,
        )
        resp.raise_for_status()
        return _b64decode(resp.json()["value"])

    def wait(self, keys: list[str], timeout: timedelta | None = None) -> None:
        """Blocks until every key in `keys` exists, or raises
        `RuntimeError` after `timeout` (or this store's own default
        timeout) - the exact contract `_store_based_barrier` depends on
        (it specifically catches `RuntimeError` to distinguish "still
        waiting" from a real failure, see that function's source)."""
        effective_timeout = (timeout or self._timeout).total_seconds()
        deadline = time.monotonic() + effective_timeout
        remaining = set(keys)
        while remaining:
            still_missing = set()
            for key in remaining:
                resp = self._request_with_retry(
                    lambda k=key: requests.get(f"{self._signaling_url}/kv/get", params={"key": k}, timeout=10),
                    deadline,
                )
                if resp.status_code != 200:
                    still_missing.add(key)
            remaining = still_missing
            if not remaining:
                return
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"QuicRendezvousStore.wait({keys!r}) timed out after {effective_timeout}s "
                    f"- still missing: {sorted(remaining)}"
                )
            time.sleep(_DEFAULT_POLL_INTERVAL_S)

    def delete_key(self, key: str) -> bool:
        deadline = time.monotonic() + self._timeout.total_seconds()
        resp = self._request_with_retry(
            lambda: requests.delete(f"{self._signaling_url}/kv", params={"key": key}, timeout=10),
            deadline,
        )
        resp.raise_for_status()
        return True

    def check(self, keys: list[str]) -> bool:
        """Non-blocking `wait` - `True` iff every key is already present
        right now, without polling/retrying FOR KEY PRESENCE (a
        transient network failure on the underlying HTTP call still
        gets the same short retry as everywhere else in this class -
        that's a transport hiccup, not a "key not present yet" answer)."""
        deadline = time.monotonic() + self._timeout.total_seconds()
        for key in keys:
            resp = self._request_with_retry(
                lambda k=key: requests.get(f"{self._signaling_url}/kv/get", params={"key": k}, timeout=10),
                deadline,
            )
            if resp.status_code != 200:
                return False
        return True

    def num_keys(self) -> int:
        # Not backed by the server (no `/kv/count` endpoint - nothing in
        # this project's actual usage needs it) - real torch internals
        # don't call this for rendezvous/barrier; only raise if someone
        # genuinely calls it, so the failure is clear rather than a
        # silently wrong count.
        raise NotImplementedError(
            "QuicRendezvousStore.num_keys() is not backed by the signaling "
            "server (no /kv/count endpoint) - not needed for rendezvous/"
            "barrier, add a server endpoint if a real caller needs this"
        )

    def set_timeout(self, timeout: timedelta) -> None:
        self._timeout = timeout
