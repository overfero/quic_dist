"""Signaling-only coordinator for UDP hole punching between any number of
peers, each pair getting its own synchronized punch time.

Exchanges endpoint info and a per-pair synchronized punch timestamp over
HTTP. Never touches UDP traffic and never relays anything.

Also exposes a small generic key-value store (`/kv/*`, added for
`quic_dist.store.QuicRendezvousStore` - see that module's docstring) -
purely additive, does not touch or depend on the hole-punch endpoints
above. Backs `torch.distributed`'s own store-based rendezvous/barrier
(`torch.distributed.distributed_c10d._store_based_barrier`, which needs
real `set`/`get`/`add`/`wait` semantics on arbitrary keys - not something
`/register`/`/peer/{id}` provide, those are hole-punch-specific) without
requiring direct reachability to any one rank, the same NAT-avoidance
this whole coordinator already exists for. `wait()` is implemented
client-side via polling `GET /kv/get?key=` (matching `wait_for_peer`'s
existing pattern above), not server-side blocking - consistent with this
file's own conventions, no new concurrency primitives needed here.
"""
import threading
import time
from typing import Dict, FrozenSet

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

app = FastAPI(title="UDP Hole Punch Coordinator")

SYNC_BUFFER_SECONDS = 5.0  # lead time given to both peers before they must start punching

_peers: Dict[str, dict] = {}
_start_at: Dict[FrozenSet[str], float] = {}  # keyed by {peer_id, self_id} - independent per pair

_kv: Dict[str, str] = {}
_kv_lock = threading.Lock()  # uvicorn can serve requests concurrently (threadpool) - add() must be atomic


class Registration(BaseModel):
    peer_id: str
    udp_port: int


def _client_ip(request: Request) -> str:
    # zrok (and most HTTP tunnels) terminate in front of this process, so the
    # real client address is in X-Forwarded-For; request.client.host would
    # otherwise just be the local tunnel agent.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host


@app.post("/register")
def register(reg: Registration, request: Request) -> dict:
    public_ip = _client_ip(request)  # never trust a client-supplied IP
    prev = _peers.get(reg.peer_id)
    if prev is None or prev["public_ip"] != public_ip or prev["udp_port"] != reg.udp_port:
        # New peer_id, or its endpoint changed (restart/rebind) - drop any frozen
        # sync times involving it so every pair it's part of renegotiates fresh,
        # without disturbing unrelated pairs that don't involve this peer_id.
        for pair in [p for p in _start_at if reg.peer_id in p]:
            del _start_at[pair]
    _peers[reg.peer_id] = {
        "peer_id": reg.peer_id,
        "public_ip": public_ip,
        "udp_port": reg.udp_port,
        "last_seen": time.time(),
    }
    return {"status": "ok", "public_ip": public_ip}


@app.get("/peer/{peer_id}")
def get_peer(peer_id: str, self_id: str = Query(..., description="Caller's own peer_id")) -> dict:
    peer = _peers.get(peer_id)
    if peer is None:
        raise HTTPException(status_code=404, detail="peer not registered yet")
    pair = frozenset((peer_id, self_id))
    if pair not in _start_at:
        _start_at[pair] = time.time() + SYNC_BUFFER_SECONDS
    return {**peer, "start_at": _start_at[pair]}


# --------------------------------------------------------------------------
# Generic key-value store - see module docstring for why this exists
# --------------------------------------------------------------------------
class KVSet(BaseModel):
    key: str
    value: str  # base64-encoded bytes - see QuicRendezvousStore, arbitrary
    # bytes (not just UTF-8 text) must round-trip through this exactly


class KVAdd(BaseModel):
    key: str
    amount: int


class KVCompareSet(BaseModel):
    key: str
    expected: str  # base64 - "" (not present) is a valid expected value
    desired: str  # base64


@app.post("/kv/set")
def kv_set(body: KVSet) -> dict:
    with _kv_lock:
        _kv[body.key] = body.value
    return {"status": "ok"}


@app.get("/kv/get")
def kv_get(key: str = Query(...)) -> dict:
    # Query param, not a path segment: torch.distributed.PrefixStore always
    # produces keys containing "/" (e.g. "mygroup/k1", "0//cpu//...") - a
    # path segment like /kv/get/{key} silently 404s on those (FastAPI
    # routing splits on "/"), which surfaced as every real barrier() call
    # hanging until timeout. A query string handles arbitrary key content.
    with _kv_lock:
        if key not in _kv:
            raise HTTPException(status_code=404, detail="key not set yet")
        return {"key": key, "value": _kv[key]}


@app.post("/kv/add")
def kv_add(body: KVAdd) -> dict:
    # torch's Store.add() stores/returns a plain integer counter, not
    # base64 bytes - kept as a separate code path from kv_set/kv_get
    # (which are always base64) rather than overloading the same key
    # namespace with two different value encodings.
    with _kv_lock:
        current = int(_kv.get(f"__counter__{body.key}", "0"))
        current += body.amount
        _kv[f"__counter__{body.key}"] = str(current)
        return {"key": body.key, "value": current}


@app.post("/kv/compare_set")
def kv_compare_set(body: KVCompareSet) -> dict:
    with _kv_lock:
        current = _kv.get(body.key, "")
        if current == body.expected:
            _kv[body.key] = body.desired
            return {"key": body.key, "value": body.desired}
        return {"key": body.key, "value": current}


@app.delete("/kv")
def kv_delete(key: str = Query(...)) -> dict:
    with _kv_lock:
        _kv.pop(key, None)
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
