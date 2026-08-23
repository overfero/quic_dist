"""Direct UDP peer-to-peer network benchmark, via NAT hole punching.

One socket, start to finish: the same UDP socket is used for STUN discovery
(mode "stun"), the synchronized hole punch, and every benchmark that follows.
No relay, no WebRTC. See README.md for how to run this against a real peer
and how to interpret every number it prints.
"""
import argparse
import asyncio
import contextlib
import itertools
import math
import os
import random
import socket
import statistics
import struct
import time
from collections import deque

import requests

try:
    import resource
    HAVE_RESOURCE = True
except ImportError:  # pragma: no cover - resource is POSIX-only
    HAVE_RESOURCE = False

KB = 1024
MB = 1024 * 1024
CHUNK_PAYLOAD = 1200  # bytes; keeps each UDP datagram under typical 1500 MTU

DEFAULT_IDLE_SECONDS = [10, 30, 60, 120]
QUICK_IDLE_SECONDS = [2, 3, 5, 8]
DEFAULT_TRANSFER_SIZES = [1 * MB, 10 * MB, 100 * MB]
QUICK_TRANSFER_SIZES = [64 * KB, 256 * KB, 1 * MB]
DEFAULT_PAYLOAD_SIZES = [256 * KB, 512 * KB, 1 * MB, 2 * MB, 4 * MB]
QUICK_PAYLOAD_SIZES = [32 * KB, 64 * KB, 128 * KB, 256 * KB, 512 * KB]
PAYLOAD_SWEEP_SIZES = [256 * KB, 512 * KB, 1 * MB, 2 * MB, 4 * MB, 8 * MB, 16 * MB, 32 * MB]
QUICK_PAYLOAD_SWEEP_SIZES = [32 * KB, 64 * KB, 128 * KB, 256 * KB, 512 * KB, 1 * MB, 2 * MB, 4 * MB]
DEFAULT_PROBE_COUNT = 200
QUICK_PROBE_COUNT = 30
PROBE_INTERVAL = 0.05
REBIND_CHECK_INTERVAL = 20.0
SOCKET_BUFFER_REQUEST = 4 * 1024 * 1024  # best-effort; kernel clamps to net.core.rmem_max/wmem_max
PACING_GAP_SECONDS = 0.002  # real sleep, not sleep(0): gives the receiver's OS process actual wall-clock time to drain
DEFAULT_STREAM_DURATION = 20.0
QUICK_STREAM_DURATION = 5.0
LONG_STREAM_DURATION = 300.0  # Experiments B/C (5 minutes)
MULTI_STREAM_COUNTS = [1, 2, 4, 8, 16]
CPU_SAMPLE_INTERVAL = 0.5

STUN_MAGIC_COOKIE = 0x2112A442


# --------------------------------------------------------------------------
# Wire protocol - tiny binary framing, one byte tag + struct-packed fields.
# --------------------------------------------------------------------------
def pack_ping(seq: int, ts: float) -> bytes:
    return b"P" + struct.pack("!Id", seq, ts)


def pack_pong(seq: int, ts: float) -> bytes:
    return b"O" + struct.pack("!Id", seq, ts)


def pack_data(stream_id: int, seq: int, payload: bytes) -> bytes:
    return b"D" + struct.pack("!BI", stream_id, seq) + payload


def pack_report_req(stream_id: int) -> bytes:
    return b"R" + struct.pack("!B", stream_id)


def pack_report_resp(stream_id: int, sr: "StreamReceiver") -> bytes:
    first_ts = sr.first_ts if sr.first_ts is not None else 0.0
    last_ts = sr.last_ts if sr.last_ts is not None else 0.0
    return b"S" + struct.pack(
        "!BIQIIIdd", stream_id, sr.received, sr.bytes, sr.duplicates, sr.out_of_order, sr.late, first_ts, last_ts
    )


def pack_whoami_req(nonce: int) -> bytes:
    return b"W" + struct.pack("!I", nonce)


def pack_whoami_resp(nonce: int, ip: str, port: int) -> bytes:
    return b"V" + struct.pack("!I", nonce) + socket.inet_aton(ip) + struct.pack("!H", port)


# --------------------------------------------------------------------------
# Minimal RFC 5389 STUN client - reuses the caller's socket, never opens
# a second one. Only a Binding Request/Response is needed.
# --------------------------------------------------------------------------
def stun_get_mapped_address(sock: socket.socket, stun_host: str, stun_port: int, timeout: float = 3.0) -> tuple[str, int]:
    txn_id = os.urandom(12)
    cookie_bytes = struct.pack("!I", STUN_MAGIC_COOKIE)
    request = struct.pack("!HH", 0x0001, 0) + cookie_bytes + txn_id
    dest = (socket.gethostbyname(stun_host), stun_port)

    sock.settimeout(timeout)
    sock.sendto(request, dest)
    data, _ = sock.recvfrom(2048)
    sock.settimeout(None)

    msg_type, msg_len = struct.unpack("!HH", data[0:4])
    if msg_type != 0x0101:
        raise RuntimeError(f"unexpected STUN response type {msg_type:#06x}")

    body = data[20:20 + msg_len]
    offset = 0
    mapped: tuple[str, int] | None = None
    while offset + 4 <= len(body):
        attr_type, attr_len = struct.unpack("!HH", body[offset:offset + 4])
        val = body[offset + 4: offset + 4 + attr_len]
        if attr_type == 0x0020 and len(val) >= 8:  # XOR-MAPPED-ADDRESS
            port = struct.unpack("!H", val[2:4])[0] ^ (STUN_MAGIC_COOKIE >> 16)
            ip_bytes = bytes(b ^ c for b, c in zip(val[4:8], cookie_bytes))
            mapped = (socket.inet_ntoa(ip_bytes), port)
            break
        if attr_type == 0x0001 and len(val) >= 8 and mapped is None:  # MAPPED-ADDRESS fallback
            port = struct.unpack("!H", val[2:4])[0]
            mapped = (socket.inet_ntoa(val[4:8]), port)
        offset += 4 + attr_len + ((4 - attr_len % 4) % 4)

    if mapped is None:
        raise RuntimeError("STUN response contained no mapped address")
    return mapped


# --------------------------------------------------------------------------
# Signaling client
# --------------------------------------------------------------------------
def other_id(self_id: str) -> str:
    return "B" if self_id.upper() == "A" else "A"


def register(signaling_url: str, peer_id: str, udp_port: int, retries: int = 10, target_id: str | None = None) -> dict:
    """`target_id`: the specific peer this registration is FOR - pass it
    whenever `peer_id` might register more than once concurrently for
    different peers (e.g. a pipeline-parallel middle rank connecting to
    both its predecessor and successor under one identity). Without it,
    the signaling server can't tell two of this identity's concurrent
    registrations apart and a later one silently overwrites an earlier
    one still in use - a real bug found via direct testing, see
    signaling_server.py's `Registration.target_id` docstring for the
    full story. `None` (the default) keeps the original single-pairing
    behavior for any caller that doesn't need this."""
    for _ in range(retries):
        try:
            resp = requests.post(
                f"{signaling_url}/register",
                json={"peer_id": peer_id, "udp_port": udp_port, "target_id": target_id},
                timeout=5,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            print(f"register failed ({exc}); retrying...")
            time.sleep(2)
    raise RuntimeError("could not reach signaling server to register")


def wait_for_peer(signaling_url: str, self_id: str, peer_id: str, poll_interval: float = 1.0) -> dict:
    while True:
        try:
            resp = requests.get(f"{signaling_url}/peer/{peer_id}", params={"self_id": self_id}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("start_at") is not None:
                    return data
        except requests.exceptions.RequestException as exc:
            print(f"poll failed ({exc}); retrying...")
        time.sleep(poll_interval)


# --------------------------------------------------------------------------
# Stats helpers
# --------------------------------------------------------------------------
def percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return s[int(k)]
    return s[f] + (s[c] - s[f]) * (k - f)


def jitter_ms(rtts: list[float]) -> float:
    if len(rtts) < 2:
        return 0.0
    diffs = [abs(rtts[i] - rtts[i - 1]) for i in range(1, len(rtts))]
    return sum(diffs) / len(diffs)


def compute_bdp(throughput_mbps: float, rtt_ms: float) -> dict:
    bandwidth_bps = throughput_mbps * 1_000_000
    rtt_s = rtt_ms / 1000
    bdp_bytes = bandwidth_bps * rtt_s / 8
    bdp_packets = bdp_bytes / CHUNK_PAYLOAD
    return {
        "bandwidth_mbps": throughput_mbps,
        "rtt_ms": rtt_ms,
        "bdp_bytes": bdp_bytes,
        "recommended_window_packets": max(1, math.ceil(bdp_packets)),
    }


def timing_snapshot(protocol: "PeerProtocol") -> dict:
    return {
        "sending": protocol.time_sending,
        "receiving": protocol.time_receiving,
        "serializing": protocol.time_serializing,
    }


def timing_delta(before: dict, after: dict, wall_elapsed: float) -> dict:
    sending = after["sending"] - before["sending"]
    receiving = after["receiving"] - before["receiving"]
    serializing = after["serializing"] - before["serializing"]
    idle = max(0.0, wall_elapsed - sending - receiving - serializing)
    denom = wall_elapsed if wall_elapsed else 1.0
    return {
        "sending_s": sending, "receiving_s": receiving, "serializing_s": serializing, "idle_s": idle,
        "sending_pct": 100 * sending / denom, "receiving_pct": 100 * receiving / denom,
        "serializing_pct": 100 * serializing / denom, "idle_pct": 100 * idle / denom,
    }


class StreamReceiver:
    """Tracks arrivals for one stream/transfer id: loss, duplicates, reordering,
    lateness, and byte/packet counts - all from what actually showed up here,
    no trust placed in whatever the sender claims it sent."""

    def __init__(self, window: int = 8192):
        self.highest = -1
        self.received = 0
        self.bytes = 0
        self.duplicates = 0
        self.out_of_order = 0
        self.late = 0
        self.recent: deque = deque(maxlen=window)
        self.recent_set: set = set()
        self.first_ts: float | None = None
        self.last_ts: float | None = None

    def observe(self, seq: int, nbytes: int) -> None:
        now = time.monotonic()
        if self.first_ts is None:
            self.first_ts = now
        self.last_ts = now

        if seq in self.recent_set:
            self.duplicates += 1
            return
        if len(self.recent) == self.recent.maxlen:
            old = self.recent.popleft()
            self.recent_set.discard(old)
        self.recent.append(seq)
        self.recent_set.add(seq)

        self.received += 1
        self.bytes += nbytes
        if seq > self.highest:
            # Extends the high-water mark: this is the normal, in-order case,
            # regardless of any earlier gap. A gap is loss (see loss_pct()),
            # not reordering - conflating the two was a real bug: once any
            # single packet was lost, every later (perfectly in-order) packet
            # used to get miscounted as "out of order" forever.
            self.highest = seq
        else:
            # A higher sequence number already arrived before this one - genuine
            # reordering. If it's arriving long after the window moved on, it's
            # too old to say anything meaningful about (could even be a duplicate
            # of something already evicted) - call that "late" instead.
            gap = self.highest - seq
            if gap > self.recent.maxlen:
                self.late += 1
            else:
                self.out_of_order += 1

    def loss_pct(self) -> float:
        total = self.highest + 1
        return 100.0 * (1 - self.received / total) if total else 0.0


class CPUSampler:
    """Periodically samples this process's own CPU time (via getrusage) to
    report user/system/overall/peak percentages over a benchmark phase."""

    def __init__(self, interval: float = CPU_SAMPLE_INTERVAL):
        self.interval = interval
        self.samples: list[tuple[float, float, float]] = []
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:
        if not HAVE_RESOURCE:
            return
        last = resource.getrusage(resource.RUSAGE_SELF)
        last_t = time.monotonic()
        while True:
            await asyncio.sleep(self.interval)
            now = resource.getrusage(resource.RUSAGE_SELF)
            now_t = time.monotonic()
            dt = now_t - last_t
            if dt > 0:
                user_pct = 100 * (now.ru_utime - last.ru_utime) / dt
                sys_pct = 100 * (now.ru_stime - last.ru_stime) / dt
                self.samples.append((user_pct, sys_pct, user_pct + sys_pct))
            last, last_t = now, now_t

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> dict:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        if not self.samples:
            return {"user_pct": 0.0, "system_pct": 0.0, "overall_pct": 0.0, "peak_pct": 0.0, "available": HAVE_RESOURCE}
        return {
            "user_pct": statistics.mean(s[0] for s in self.samples),
            "system_pct": statistics.mean(s[1] for s in self.samples),
            "overall_pct": statistics.mean(s[2] for s in self.samples),
            "peak_pct": max(s[2] for s in self.samples),
            "available": True,
        }


# --------------------------------------------------------------------------
# Protocol / transport
# --------------------------------------------------------------------------
class PeerProtocol(asyncio.DatagramProtocol):
    def __init__(self, peer_addr: tuple[str, int]):
        self.peer_addr = peer_addr
        self.transport: asyncio.DatagramTransport | None = None
        self.loop = asyncio.get_event_loop()
        self.established = False
        self.ping_waiters: dict[int, asyncio.Future] = {}
        self.whoami_waiters: dict[int, asyncio.Future] = {}
        self.report_waiters: dict[int, asyncio.Future] = {}
        self.stream_receivers: dict[int, StreamReceiver] = {}
        self.barrier_seen: set[bytes] = set()
        self.pacing_burst = 64  # packets sent before a real pacing sleep; sized to the granted socket buffer in main()
        self.peer_done = False
        self.rebind_event_count = 0
        # The real socket, for --fast-path sends. transport.get_extra_info("socket")
        # deliberately returns a restricted TransportSocket wrapper with no sendto() -
        # asyncio's guard against exactly this kind of bypass - so main() sets this
        # directly from the socket object it created, before handing it to asyncio.
        self.raw_sock: socket.socket | None = None
        # Timing breakdown counters (cumulative; callers snapshot/diff around a phase).
        self.time_sending = 0.0
        self.time_receiving = 0.0
        self.time_serializing = 0.0

    def connection_made(self, transport: asyncio.DatagramTransport) -> None:
        self.transport = transport

    def send(self, data: bytes) -> None:
        t0 = time.perf_counter()
        self.transport.sendto(data, self.peer_addr)
        self.time_sending += time.perf_counter() - t0

    def datagram_received(self, data: bytes, addr) -> None:
        t0 = time.perf_counter()
        try:
            self._handle_datagram(data, addr)
        finally:
            self.time_receiving += time.perf_counter() - t0

    def _handle_datagram(self, data: bytes, addr) -> None:
        if self.established and addr != self.peer_addr:
            print(f"NAT REBINDING DETECTED (peer side): {self.peer_addr} -> {addr}")
            self.rebind_event_count += 1
        self.peer_addr = addr

        if not self.established:
            self.established = True
            print("Hole punch success.")

        if not data:
            return
        tag = data[0:1]
        try:
            if tag == b"P":
                seq, ts = struct.unpack("!Id", data[1:13])
                self.send(pack_pong(seq, ts))
            elif tag == b"O":
                seq, ts = struct.unpack("!Id", data[1:13])
                fut = self.ping_waiters.get(seq)
                if fut and not fut.done():
                    fut.set_result(True)
            elif tag == b"D":
                stream_id, seq = struct.unpack("!BI", data[1:6])
                payload = data[6:]
                sr = self.stream_receivers.get(stream_id)
                if sr is None:
                    sr = self.stream_receivers[stream_id] = StreamReceiver()
                sr.observe(seq, len(payload))
            elif tag == b"R":
                (stream_id,) = struct.unpack("!B", data[1:2])
                sr = self.stream_receivers.get(stream_id, StreamReceiver())
                self.send(pack_report_resp(stream_id, sr))
            elif tag == b"S":
                stream_id, count, nbytes, dup, ooo, late, first_ts, last_ts = struct.unpack("!BIQIIIdd", data[1:42])
                fut = self.report_waiters.get(stream_id)
                if fut and not fut.done():
                    fut.set_result({
                        "count": count, "bytes": nbytes, "duplicates": dup, "out_of_order": ooo, "late": late,
                        "first_ts": first_ts or None, "last_ts": last_ts or None,
                    })
            elif tag == b"W":
                (nonce,) = struct.unpack("!I", data[1:5])
                self.send(pack_whoami_resp(nonce, addr[0], addr[1]))
            elif tag == b"V":
                (nonce,) = struct.unpack("!I", data[1:5])
                ip = socket.inet_ntoa(data[5:9])
                port = struct.unpack("!H", data[9:11])[0]
                fut = self.whoami_waiters.get(nonce)
                if fut and not fut.done():
                    fut.set_result((ip, port))
            elif tag == b"F":
                self.peer_done = True
            elif tag == b"X":
                self.barrier_seen.add(data[1:])
        except (struct.error, IndexError):
            return  # malformed/unexpected packet - ignore, never crash the benchmark

    async def ping(self, seq: int, timeout: float) -> float | None:
        fut = self.loop.create_future()
        self.ping_waiters[seq] = fut
        t0 = time.monotonic()
        self.send(pack_ping(seq, t0))
        try:
            await asyncio.wait_for(fut, timeout)
            return (time.monotonic() - t0) * 1000
        except asyncio.TimeoutError:
            return None
        finally:
            self.ping_waiters.pop(seq, None)

    async def whoami(self, nonce: int, timeout: float) -> tuple[str, int] | None:
        fut = self.loop.create_future()
        self.whoami_waiters[nonce] = fut
        self.send(pack_whoami_req(nonce))
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            return None
        finally:
            self.whoami_waiters.pop(nonce, None)

    async def request_report(self, stream_id: int, attempts: int = 5, per_attempt_timeout: float = 1.5) -> dict | None:
        for _ in range(attempts):
            fut = self.loop.create_future()
            self.report_waiters[stream_id] = fut
            self.send(pack_report_req(stream_id))
            try:
                return await asyncio.wait_for(fut, per_attempt_timeout)
            except asyncio.TimeoutError:
                continue
            finally:
                self.report_waiters.pop(stream_id, None)
        return None


async def barrier(protocol: PeerProtocol, marker: bytes, timeout: float = 60.0, interval: float = 0.5) -> bool:
    """Rendezvous point: block until the peer has also reached this same
    checkpoint (or timeout). Used to align phase transitions between the two
    independent peer processes instead of relying on incidental timing."""
    protocol.barrier_seen.discard(marker)
    deadline = time.monotonic() + timeout
    while marker not in protocol.barrier_seen and time.monotonic() < deadline:
        protocol.send(b"X" + marker)
        await asyncio.sleep(interval)
    reached = marker in protocol.barrier_seen
    for _ in range(3):  # make sure our own arrival is seen even if the peer already moved on
        protocol.send(b"X" + marker)
        await asyncio.sleep(0.1)
    return reached


async def punch_loop(protocol: PeerProtocol, interval: float = 0.1) -> None:
    seq = 0
    while True:
        protocol.send(pack_ping(seq, time.monotonic()))
        seq += 1
        await asyncio.sleep(interval)


async def rebind_watcher(protocol: PeerProtocol, interval: float) -> None:
    nonces = itertools.count(1)
    known: tuple[str, int] | None = None
    while True:
        await asyncio.sleep(interval)
        observed = await protocol.whoami(next(nonces), timeout=3.0)
        if observed is None:
            print("[rebind-check] no response (packet lost?) - skipping this round")
            continue
        if known is not None and observed != known:
            print(f"NAT REBINDING DETECTED (own side): {known[0]}:{known[1]} -> {observed[0]}:{observed[1]}")
            protocol.rebind_event_count += 1
        known = observed


# --------------------------------------------------------------------------
# Streaming primitives - shared by continuous, duplex, and multi-stream tests
# --------------------------------------------------------------------------
async def sample_throughput(get_bytes_fn, duration: float, interval: float = 1.0) -> dict:
    """Samples a monotonically-increasing byte counter at intervals to report
    avg/peak/min instantaneous throughput over the window, not just a single
    start-to-end average that would hide bursts and stalls."""
    step = min(interval, max(0.1, duration / 10))
    samples = []
    start = time.monotonic()
    last_bytes = get_bytes_fn()
    last_t = start
    first = True
    while time.monotonic() - start < duration:
        await asyncio.sleep(step)
        now = time.monotonic()
        b = get_bytes_fn()
        dt = now - last_t
        # Skip the first interval: it partly measures barrier/scheduling skew
        # before data actually started flowing, which would otherwise show up
        # as a spuriously low "min" throughput sample.
        if dt > 0 and not first:
            samples.append((b - last_bytes) * 8 / (dt * 1_000_000))
        first = False
        last_bytes, last_t = b, now
    if not samples:
        return {"avg_mbps": 0.0, "peak_mbps": 0.0, "min_mbps": 0.0, "samples": 0}
    return {
        "avg_mbps": statistics.mean(samples), "peak_mbps": max(samples), "min_mbps": min(samples),
        "samples": len(samples),
    }


LOADED_LATENCY_SEQ_BASE = 700_000


async def run_loaded_latency(protocol: PeerProtocol, duration: float, interval: float = 0.1, timeout: float = 1.0) -> dict:
    """Pings every `interval` seconds while other traffic is in flight, so the
    resulting RTT distribution reflects latency *under load* rather than on
    an otherwise-idle link."""
    rtts: list[float] = []
    lost = 0
    seq_counter = itertools.count(LOADED_LATENCY_SEQ_BASE)
    t_start = time.monotonic()
    while time.monotonic() - t_start < duration:
        rtt = await protocol.ping(next(seq_counter), timeout)
        if rtt is None:
            lost += 1
        else:
            rtts.append(rtt)
        await asyncio.sleep(interval)
    total = lost + len(rtts)
    return {
        "avg_rtt_ms": statistics.mean(rtts) if rtts else float("nan"),
        "p95_rtt_ms": percentile(rtts, 95),
        "p99_rtt_ms": percentile(rtts, 99),
        "jitter_ms": jitter_ms(rtts),
        "loss_pct": 100.0 * lost / total if total else 0.0,
        "samples": len(rtts),
    }


async def stream_send(protocol: PeerProtocol, stream_id: int, duration: float,
                       pacing_mbps: float | None = None, pacing_pps: float | None = None,
                       fast_path: bool = False) -> dict:
    """Sends fixed-size DATA chunks tagged `stream_id` continuously for
    `duration` seconds. `fast_path` preallocates one mutable buffer and
    patches only the sequence number in place + calls the raw socket
    directly, to isolate Python-level per-packet overhead from whatever the
    normal path (pack_data() allocation + protocol.send()) costs."""
    filler = os.urandom(CHUNK_PAYLOAD)
    seq = 0
    sent_bytes = 0
    rate_samples: list[float] = []
    t_start = time.monotonic()
    last_sample_t, last_sample_bytes = t_start, 0

    raw_sock = protocol.raw_sock if fast_path else None
    buf = None
    if fast_path:
        buf = bytearray(6 + CHUNK_PAYLOAD)
        buf[0:1] = b"D"
        buf[1] = stream_id
        buf[6:] = filler

    target_interval = None
    if pacing_pps:
        target_interval = 1.0 / pacing_pps
    elif pacing_mbps:
        pkt_bits = (CHUNK_PAYLOAD + 6) * 8
        target_interval = pkt_bits / (pacing_mbps * 1_000_000)
    next_send_time = t_start

    while time.monotonic() - t_start < duration:
        if fast_path:
            struct.pack_into("!I", buf, 2, seq)
            try:
                raw_sock.sendto(buf, protocol.peer_addr)
            except BlockingIOError:
                pass  # OS send buffer momentarily full at max speed - drop and keep going
        else:
            t0 = time.perf_counter()
            packet = pack_data(stream_id, seq, filler)
            protocol.time_serializing += time.perf_counter() - t0
            protocol.send(packet)
        sent_bytes += CHUNK_PAYLOAD
        seq += 1

        now = time.monotonic()
        if now - last_sample_t >= 1.0:
            dt = now - last_sample_t
            rate_samples.append((sent_bytes - last_sample_bytes) * 8 / (dt * 1_000_000))
            last_sample_bytes, last_sample_t = sent_bytes, now

        if target_interval is not None:
            next_send_time += target_interval
            sleep_for = next_send_time - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
        elif seq % protocol.pacing_burst == 0:
            await asyncio.sleep(PACING_GAP_SECONDS)

    elapsed = time.monotonic() - t_start
    return {
        "packets_sent": seq,
        "bytes_sent": sent_bytes,
        "elapsed_s": elapsed,
        "avg_send_mbps": (sent_bytes * 8) / (elapsed * 1_000_000) if elapsed > 0 else 0.0,
        "peak_send_mbps": max(rate_samples) if rate_samples else 0.0,
        "min_send_mbps": min(rate_samples) if rate_samples else 0.0,
        "fast_path": fast_path,
    }


async def run_streaming_test(protocol: PeerProtocol, stream_id: int, duration: float, direction: str,
                              pacing_mbps: float | None = None, pacing_pps: float | None = None,
                              fast_path: bool = False, measure_loaded_latency: bool = True,
                              measure_cpu: bool = True) -> dict:
    """direction: 'send' (this peer only sends), 'recv' (only receives), or
    'duplex' (both, simultaneously). Both peers must call this with the same
    stream_id; the barrier() calls keep them aligned regardless of local
    timing drift from earlier phases."""
    do_send = direction in ("send", "duplex")
    do_recv = direction in ("recv", "duplex")
    marker = f"STREAM{stream_id}".encode()

    if do_recv:
        protocol.stream_receivers[stream_id] = StreamReceiver()

    cpu = CPUSampler() if measure_cpu else None
    await barrier(protocol, marker + b"S")
    if cpu:
        cpu.start()

    tasks = {}
    if do_send:
        tasks["send"] = asyncio.create_task(
            stream_send(protocol, stream_id, duration, pacing_mbps, pacing_pps, fast_path))
    if do_recv:
        tasks["recv"] = asyncio.create_task(
            sample_throughput(lambda: protocol.stream_receivers[stream_id].bytes, duration))
    if measure_loaded_latency:
        tasks["latency"] = asyncio.create_task(run_loaded_latency(protocol, duration))

    values = await asyncio.gather(*tasks.values())
    results = dict(zip(tasks.keys(), values))

    cpu_stats = await cpu.stop() if cpu else None
    await barrier(protocol, marker + b"E")

    out: dict = {"direction": direction, "duration_s": duration, "stream_id": stream_id}
    if cpu_stats:
        out["cpu"] = cpu_stats
    if do_send:
        out["send"] = results["send"]
        out["peer_received_from_me"] = await protocol.request_report(stream_id)
    if do_recv:
        # Don't pop/delete here: the sender's request_report() call (a separate
        # process, on its own schedule) may not have asked for this stream's
        # final tally yet. Deleting eagerly raced against that and made the
        # sender see a phantom empty report. Entries are cheap enough to just
        # leave for the life of the process.
        sr = protocol.stream_receivers.get(stream_id)
        recv_summary = results["recv"]
        if sr:
            recv_summary = {
                "bytes": sr.bytes, "packets": sr.received, "loss_pct": sr.loss_pct(),
                "duplicates": sr.duplicates, "out_of_order": sr.out_of_order, "late": sr.late,
                **recv_summary,
            }
        out["receive"] = recv_summary
    if measure_loaded_latency:
        out["loaded_latency"] = results["latency"]
    return out


async def run_multi_stream(protocol: PeerProtocol, base_stream_id: int, n_streams: int, duration: float,
                            pacing_mbps: float | None = None, pacing_pps: float | None = None,
                            fast_path: bool = False) -> dict:
    """Runs n_streams concurrent duplex flows over the SAME single socket
    (distinguished only by stream_id), to see whether aggregate achievable
    throughput scales with concurrency - it won't if a single-threaded
    Python event loop's per-packet overhead is the real ceiling."""
    cpu = CPUSampler()
    cpu.start()
    per_stream = await asyncio.gather(*[
        run_streaming_test(protocol, base_stream_id + i, duration, "duplex",
                            pacing_mbps=pacing_mbps, pacing_pps=pacing_pps, fast_path=fast_path,
                            measure_loaded_latency=(i == 0), measure_cpu=False)
        for i in range(n_streams)
    ])
    cpu_stats = await cpu.stop()
    total_send = sum(r["send"]["avg_send_mbps"] for r in per_stream)
    total_recv = sum(r["receive"]["avg_mbps"] for r in per_stream)
    return {
        "n_streams": n_streams,
        "aggregate_send_mbps": total_send,
        "aggregate_recv_mbps": total_recv,
        "cpu": cpu_stats,
        "per_stream": per_stream,
    }


# --------------------------------------------------------------------------
# Core benchmarks
# --------------------------------------------------------------------------
async def run_probe_benchmark(protocol: PeerProtocol, count: int, interval: float, timeout: float = 2.0) -> dict:
    rtts: list[float] = []
    lost = 0
    for seq in range(count):
        rtt = await protocol.ping(seq, timeout)
        if rtt is None:
            lost += 1
        else:
            rtts.append(rtt)
            if seq % 20 == 0:
                print(f"Current RTT: {rtt:.2f} ms")
        await asyncio.sleep(interval)
    return {
        "sent": count,
        "received": len(rtts),
        "loss_pct": 100.0 * lost / count if count else 0.0,
        "avg_rtt_ms": statistics.mean(rtts) if rtts else float("nan"),
        "p95_rtt_ms": percentile(rtts, 95),
        "p99_rtt_ms": percentile(rtts, 99),
        "jitter_ms": jitter_ms(rtts),
    }


async def run_transfer(protocol: PeerProtocol, stream_id: int, size_bytes: int) -> dict:
    """One-shot fixed-size transfer: send `size_bytes` as CHUNK_PAYLOAD-sized
    packets, then ask the receiver what it actually got. Timing does not
    stop until that confirmation arrives - "throughput" here means bytes
    confirmed delivered per second of true end-to-end elapsed time, not how
    fast this process could shove data into a socket."""
    total_chunks = math.ceil(size_bytes / CHUNK_PAYLOAD)
    filler = os.urandom(CHUNK_PAYLOAD)
    burst = protocol.pacing_burst
    t_start = time.monotonic()
    for seq in range(total_chunks):
        remaining = size_bytes - seq * CHUNK_PAYLOAD
        payload = filler if remaining >= CHUNK_PAYLOAD else filler[:remaining]
        t0 = time.perf_counter()
        packet = pack_data(stream_id, seq, payload)
        protocol.time_serializing += time.perf_counter() - t0
        protocol.send(packet)
        if seq % burst == burst - 1:
            # A real sleep (not sleep(0)) actually yields the CPU to the OS scheduler,
            # giving the receiving process wall-clock time to drain its socket buffer.
            # Sending faster than the receiver can drain just overflows the kernel's
            # UDP receive buffer and looks like "loss" that never touched the network.
            await asyncio.sleep(PACING_GAP_SECONDS)
    t_sent = time.monotonic()

    report = await protocol.request_report(stream_id)
    t_done = time.monotonic()  # timing stops only once the receiver has confirmed what it actually got

    received = report["count"] if report else 0
    recv_bytes = report["bytes"] if report else 0
    elapsed = max(t_done - t_start, 1e-6)
    reassembly_ms = None
    if report and report.get("first_ts") and report.get("last_ts"):
        reassembly_ms = (report["last_ts"] - report["first_ts"]) * 1000
    return {
        "size_bytes": size_bytes,
        "chunks_sent": total_chunks,
        "chunks_received": received,
        "loss_pct": 100.0 * (1 - received / total_chunks) if total_chunks else 0.0,
        "throughput_mbps": (recv_bytes * 8) / (elapsed * 1_000_000),
        "send_phase_s": t_sent - t_start,
        "completion_latency_ms": (t_done - t_start) * 1000,
        "duplicates": report["duplicates"] if report else 0,
        "out_of_order": report["out_of_order"] if report else 0,
        "late": report["late"] if report else 0,
        "reassembly_time_ms": reassembly_ms,
    }


async def run_keepalive_test(protocol: PeerProtocol, idle_seconds_list: list[int], attempts: int = 3) -> list[dict]:
    results = []
    for idle in idle_seconds_list:
        print(f"\nIdling for {idle}s (NAT keepalive test)...")
        await asyncio.sleep(idle)
        # A single dropped probe isn't proof the mapping expired, so retry a
        # few times before concluding the NAT actually closed the hole.
        rtt = None
        for attempt in range(attempts):
            rtt = await protocol.ping(900_000 + idle * 10 + attempt, timeout=1.5)
            if rtt is not None:
                break
        survived = rtt is not None
        detail = f" (rtt={rtt:.1f} ms)" if survived else ""
        print(f"NAT timeout result ({idle}s idle): {'SURVIVED' if survived else 'MAPPING EXPIRED / NO RESPONSE'}{detail}")
        results.append({"idle_seconds": idle, "survived": survived, "rtt_ms": rtt})
    return results


def _print_transfer_line(r: dict, unit_divisor: float, unit_label: str) -> None:
    reasm = f", reassembly={r['reassembly_time_ms']:.1f}ms" if r.get("reassembly_time_ms") is not None else ""
    print(f"  {r['size_bytes'] / unit_divisor:9.0f} {unit_label} -> {r['throughput_mbps']:8.2f} Mbps "
          f"loss={r['loss_pct']:5.1f}% latency={r['completion_latency_ms']:8.1f}ms "
          f"pkts={r['chunks_sent']} dup={r['duplicates']} ooo={r['out_of_order']}{reasm}")


def _print_stream_result(s: dict | None) -> None:
    if not s:
        return
    direction = s.get("direction", "?")
    line = f"  [{direction}] duration={s['duration_s']:.0f}s"
    if "send" in s:
        sd = s["send"]
        line += f"  send: avg={sd['avg_send_mbps']:.2f} peak={sd['peak_send_mbps']:.2f} min={sd['min_send_mbps']:.2f} Mbps"
        pr = s.get("peer_received_from_me")
        if pr:
            line += f" (peer confirmed {pr['bytes'] / MB:.2f} MB, {pr.get('duplicates', 0)} dup)"
    if "receive" in s:
        rc = s["receive"]
        line += (f"  recv: avg={rc['avg_mbps']:.2f} peak={rc['peak_mbps']:.2f} min={rc['min_mbps']:.2f} Mbps, "
                 f"loss={rc['loss_pct']:.1f}%, dup={rc['duplicates']} ooo={rc['out_of_order']} late={rc['late']}")
    print(line)
    if "loaded_latency" in s:
        ll = s["loaded_latency"]
        print(f"      loaded RTT: avg={ll['avg_rtt_ms']:.2f}ms P95={ll['p95_rtt_ms']:.2f}ms "
              f"P99={ll['p99_rtt_ms']:.2f}ms jitter={ll['jitter_ms']:.2f}ms loss={ll['loss_pct']:.1f}%")
    if s.get("cpu"):
        c = s["cpu"]
        print(f"      CPU: user={c['user_pct']:.1f}% sys={c['system_pct']:.1f}% "
              f"overall={c['overall_pct']:.1f}% peak={c['peak_pct']:.1f}%")


# --------------------------------------------------------------------------
# Experiments - each populates `report` and prints as it goes
# --------------------------------------------------------------------------
async def run_default_experiment(protocol: PeerProtocol, report: dict, tid_counter, transfer_sizes: list[int],
                                  payload_sizes: list[int], args, is_leader: bool,
                                  pacing_mbps: float | None, pacing_pps: float | None) -> None:
    transfer_results = []
    print("Running throughput benchmark (synchronized start; both directions overlap by design)...")
    timing_before = timing_snapshot(protocol)
    t0 = time.monotonic()
    for size in transfer_sizes:
        stream_id = next(tid_counter)
        await barrier(protocol, f"XFER{stream_id}".encode())
        result = await run_transfer(protocol, stream_id, size)
        transfer_results.append(result)
        _print_transfer_line(result, MB if size >= MB else KB, "MB" if size >= MB else "KB")
    report["transfers"] = transfer_results
    report["timing_breakdown"] = timing_delta(timing_before, timing_snapshot(protocol), time.monotonic() - t0)

    payload_results = []
    print("\nRunning payload benchmark (hidden-state-sized payloads)...")
    for size in payload_sizes:
        stream_id = next(tid_counter)
        await barrier(protocol, f"XFER{stream_id}".encode())
        result = await run_transfer(protocol, stream_id, size)
        payload_results.append(result)
        _print_transfer_line(result, KB, "KB")
    report["payloads"] = payload_results

    duration = args.stream_duration or (QUICK_STREAM_DURATION if args.quick else DEFAULT_STREAM_DURATION)
    stream_id_a, stream_id_b = next(tid_counter), next(tid_counter)
    print(f"\nRunning uni-directional streaming ({duration:.0f}s each direction, leader={is_leader})...")
    phase1 = await run_streaming_test(protocol, stream_id_a, duration, "send" if is_leader else "recv",
                                       pacing_mbps=pacing_mbps, pacing_pps=pacing_pps, fast_path=args.fast_path,
                                       measure_loaded_latency=False, measure_cpu=False)
    phase2 = await run_streaming_test(protocol, stream_id_b, duration, "recv" if is_leader else "send",
                                       pacing_mbps=pacing_mbps, pacing_pps=pacing_pps, fast_path=args.fast_path,
                                       measure_loaded_latency=False, measure_cpu=False)
    report["uni_streams"] = [phase1, phase2]
    for p in (phase1, phase2):
        _print_stream_result(p)

    duplex_id = next(tid_counter)
    print(f"\nRunning duplex streaming with loaded-latency probes ({duration:.0f}s)...")
    duplex_result = await run_streaming_test(protocol, duplex_id, duration, "duplex",
                                              pacing_mbps=pacing_mbps, pacing_pps=pacing_pps, fast_path=args.fast_path)
    report["duplex"] = duplex_result
    _print_stream_result(duplex_result)
    report["cpu"] = duplex_result.get("cpu")

    candidate_throughputs = [r["throughput_mbps"] for r in transfer_results + payload_results]
    candidate_throughputs.append(duplex_result["receive"]["avg_mbps"])
    best_throughput = max(candidate_throughputs, default=0.0)
    report["bdp"] = compute_bdp(best_throughput, report["probe"]["avg_rtt_ms"])


async def run_payload_sweep_experiment(protocol: PeerProtocol, report: dict, tid_counter, args) -> None:
    sizes = QUICK_PAYLOAD_SWEEP_SIZES if args.quick else PAYLOAD_SWEEP_SIZES
    results = []
    print(f"Running payload sweep ({len(sizes)} sizes, 256KB-32MB range) (Experiment A)...")
    for size in sizes:
        stream_id = next(tid_counter)
        await barrier(protocol, f"XFER{stream_id}".encode())
        result = await run_transfer(protocol, stream_id, size)
        results.append(result)
        _print_transfer_line(result, KB, "KB")
    report["payload_sweep"] = results
    if results:
        report["bdp"] = compute_bdp(max(r["throughput_mbps"] for r in results), report["probe"]["avg_rtt_ms"])


async def run_long_streaming_experiment(protocol: PeerProtocol, report: dict, tid_counter, args, is_leader: bool,
                                         pacing_mbps: float | None, pacing_pps: float | None) -> None:
    duration = args.stream_duration or LONG_STREAM_DURATION
    stream_id_a, stream_id_b = next(tid_counter), next(tid_counter)
    print(f"Running uni-directional streaming for {duration:.0f}s each direction (Experiment B)...")
    phase1 = await run_streaming_test(protocol, stream_id_a, duration, "send" if is_leader else "recv",
                                       pacing_mbps=pacing_mbps, pacing_pps=pacing_pps, fast_path=args.fast_path)
    phase2 = await run_streaming_test(protocol, stream_id_b, duration, "recv" if is_leader else "send",
                                       pacing_mbps=pacing_mbps, pacing_pps=pacing_pps, fast_path=args.fast_path)
    report["uni_streams"] = [phase1, phase2]
    for p in (phase1, phase2):
        _print_stream_result(p)


async def run_long_duplex_experiment(protocol: PeerProtocol, report: dict, tid_counter, args,
                                      pacing_mbps: float | None, pacing_pps: float | None) -> None:
    duration = args.stream_duration or LONG_STREAM_DURATION
    stream_id = next(tid_counter)
    print(f"Running duplex streaming for {duration:.0f}s (Experiment C)...")
    result = await run_streaming_test(protocol, stream_id, duration, "duplex",
                                       pacing_mbps=pacing_mbps, pacing_pps=pacing_pps, fast_path=args.fast_path)
    report["duplex"] = result
    _print_stream_result(result)


async def run_loaded_latency_experiment(protocol: PeerProtocol, report: dict, tid_counter, args,
                                         pacing_mbps: float | None, pacing_pps: float | None) -> None:
    duration = args.stream_duration or (QUICK_STREAM_DURATION if args.quick else DEFAULT_STREAM_DURATION)
    stream_id = next(tid_counter)
    print(f"Running loaded-latency test ({duration:.0f}s duplex stream + 100ms ping) (Experiment D)...")
    result = await run_streaming_test(protocol, stream_id, duration, "duplex",
                                       pacing_mbps=pacing_mbps, pacing_pps=pacing_pps, fast_path=args.fast_path)
    report["loaded_latency_test"] = result
    ll = result["loaded_latency"]
    idle = report["probe"]
    print(f"  Idle RTT:   avg={idle['avg_rtt_ms']:.2f}ms P95={idle['p95_rtt_ms']:.2f}ms P99={idle['p99_rtt_ms']:.2f}ms")
    print(f"  Loaded RTT: avg={ll['avg_rtt_ms']:.2f}ms P95={ll['p95_rtt_ms']:.2f}ms P99={ll['p99_rtt_ms']:.2f}ms "
          f"jitter={ll['jitter_ms']:.2f}ms")


async def run_multi_stream_experiment(protocol: PeerProtocol, report: dict, tid_counter, args,
                                       pacing_mbps: float | None, pacing_pps: float | None) -> None:
    duration = args.stream_duration or (QUICK_STREAM_DURATION if args.quick else 15.0)
    counts = [1, 2] if args.quick else MULTI_STREAM_COUNTS
    results = []
    print(f"Running multi-stream scaling test ({counts}, {duration:.0f}s each) (Experiment E)...")
    for n in counts:
        base_id = next(tid_counter)
        for _ in range(n - 1):
            next(tid_counter)
        result = await run_multi_stream(protocol, base_id, n, duration, pacing_mbps, pacing_pps, args.fast_path)
        results.append(result)
        print(f"  {n:2d} streams -> aggregate send {result['aggregate_send_mbps']:8.2f} Mbps, "
              f"aggregate recv {result['aggregate_recv_mbps']:8.2f} Mbps, CPU {result['cpu']['overall_pct']:.1f}%")
    report["multi_stream"] = results


async def run_fast_path_compare_experiment(protocol: PeerProtocol, report: dict, tid_counter, args,
                                            pacing_mbps: float | None, pacing_pps: float | None) -> None:
    duration = args.stream_duration or (QUICK_STREAM_DURATION if args.quick else 20.0)
    print(f"Running Python-overhead comparison: normal vs fast-path ({duration:.0f}s each) (Experiment F)...")
    normal_id = next(tid_counter)
    normal = await run_streaming_test(protocol, normal_id, duration, "duplex",
                                       pacing_mbps=pacing_mbps, pacing_pps=pacing_pps, fast_path=False)
    fast_id = next(tid_counter)
    fast = await run_streaming_test(protocol, fast_id, duration, "duplex",
                                     pacing_mbps=pacing_mbps, pacing_pps=pacing_pps, fast_path=True)
    report["fast_path_compare"] = {"normal": normal, "fast_path": fast}
    print(f"  normal:    send {normal['send']['avg_send_mbps']:.2f} Mbps, recv {normal['receive']['avg_mbps']:.2f} Mbps")
    print(f"  fast-path: send {fast['send']['avg_send_mbps']:.2f} Mbps, recv {fast['receive']['avg_mbps']:.2f} Mbps")


# --------------------------------------------------------------------------
# Final report
# --------------------------------------------------------------------------
def print_final_report(report: dict) -> None:
    print("\n" + "=" * 70)
    print("FINAL BENCHMARK REPORT")
    print("=" * 70)
    print(f"Hole punching: {'SUCCESS' if report.get('hole_punch_success') else 'FAILED'}")
    own, peer = report.get("own_addr"), report.get("peer_addr")
    if own and peer:
        print(f"External endpoint: {own[0]}:{own[1]}")
        print(f"Peer endpoint:     {peer[0]}:{peer[1]}")
    if report.get("direct_p2p_verified") is not None:
        print(f"Direct P2P verified (traffic origin matched coordinator-reported endpoint): "
              f"{report['direct_p2p_verified']}")
        print("  The signaling server exposes no data-plane endpoint, so this traffic cannot be relayed by it "
              "even if this check failed - a mismatch here only means the NAT remapped before the first packet.")

    buf = report.get("buffers")
    if buf:
        print(f"\nSocket buffers: requested {buf['requested']} bytes each; "
              f"kernel granted SO_RCVBUF={buf['rcvbuf']}  SO_SNDBUF={buf['sndbuf']}")

    probe = report.get("probe")
    if probe:
        print(f"\nIdle RTT: avg={probe['avg_rtt_ms']:.2f}ms  P95={probe['p95_rtt_ms']:.2f}ms  "
              f"P99={probe['p99_rtt_ms']:.2f}ms  jitter={probe['jitter_ms']:.2f}ms  loss={probe['loss_pct']:.1f}%")

    for label, xfers in (("Transfer benchmark", report.get("transfers")),
                          ("Payload benchmark", report.get("payloads")),
                          ("Payload sweep", report.get("payload_sweep"))):
        if not xfers:
            continue
        print(f"\n{label}:")
        for r in xfers:
            _print_transfer_line(r, MB if r["size_bytes"] >= MB else KB, "MB" if r["size_bytes"] >= MB else "KB")

    for label, key in (("Uni-directional streaming", "uni_streams"),
                        ("Duplex streaming", "duplex"),
                        ("Loaded-latency test", "loaded_latency_test")):
        s = report.get(key)
        if not s:
            continue
        print(f"\n{label}:")
        if isinstance(s, list):
            for item in s:
                _print_stream_result(item)
        else:
            _print_stream_result(s)

    multi = report.get("multi_stream")
    if multi:
        print("\nMulti-stream scaling:")
        for m in multi:
            print(f"  {m['n_streams']:2d} streams -> aggregate send {m['aggregate_send_mbps']:8.2f} Mbps, "
                  f"aggregate recv {m['aggregate_recv_mbps']:8.2f} Mbps  (CPU overall {m['cpu']['overall_pct']:.1f}%)")
        base = multi[0]["aggregate_recv_mbps"] or 1.0
        print(f"  scaling vs 1 stream: " +
              ", ".join(f"{m['n_streams']}x={m['aggregate_recv_mbps'] / base:.2f}x" for m in multi))

    fpc = report.get("fast_path_compare")
    if fpc:
        normal, fast = fpc["normal"], fpc["fast_path"]
        print("\nPython overhead check (normal vs fast-path):")
        print(f"  normal:    send {normal['send']['avg_send_mbps']:.2f} Mbps, "
              f"recv {normal['receive']['avg_mbps']:.2f} Mbps, CPU {normal.get('cpu', {}).get('overall_pct', 0):.1f}%")
        print(f"  fast-path: send {fast['send']['avg_send_mbps']:.2f} Mbps, "
              f"recv {fast['receive']['avg_mbps']:.2f} Mbps, CPU {fast.get('cpu', {}).get('overall_pct', 0):.1f}%")
        base_rate = normal["send"]["avg_send_mbps"] or 1e-9
        speedup = fast["send"]["avg_send_mbps"] / base_rate
        verdict = ("Python overhead looks like the throughput ceiling here (fast-path notably faster)."
                   if speedup > 1.3 else
                   "Fast-path gave little improvement - the network/receiver, not raw Python send overhead, "
                   "is the likelier ceiling.")
        print(f"  -> {speedup:.2f}x. {verdict}")

    timing = report.get("timing_breakdown")
    if timing:
        print("\nTime breakdown during throughput+payload phase:")
        print(f"  sending={timing['sending_pct']:.1f}%  receiving={timing['receiving_pct']:.1f}%  "
              f"serializing={timing['serializing_pct']:.1f}%  idle/asyncio={timing['idle_pct']:.1f}%")

    cpu = report.get("cpu")
    if cpu:
        avail_note = "" if cpu.get("available", True) else "  (resource module unavailable on this platform)"
        print(f"\nCPU utilization (duplex phase): user={cpu['user_pct']:.1f}%  system={cpu['system_pct']:.1f}%  "
              f"overall={cpu['overall_pct']:.1f}%  peak={cpu['peak_pct']:.1f}%{avail_note}")

    bdp = report.get("bdp")
    if bdp:
        print(f"\nBandwidth-delay product: {bdp['bandwidth_mbps']:.1f} Mbps x {bdp['rtt_ms']:.2f} ms RTT "
              f"-> {bdp['bdp_bytes'] / KB:.1f} KB in flight -> recommended window "
              f"~{bdp['recommended_window_packets']} packets")

    keepalive = report.get("keepalive")
    if keepalive:
        print("\nNAT keepalive:")
        for r in keepalive:
            print(f"  {r['idle_seconds']:4d}s idle -> {'survived' if r['survived'] else 'EXPIRED'}")

    print(f"\nNAT rebinding events detected: {report.get('rebind_events', 0)}")
    print("=" * 70)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
async def main() -> None:
    parser = argparse.ArgumentParser(description="Direct UDP peer-to-peer network benchmark")
    parser.add_argument("--id", required=True, help="This peer's id, e.g. A or B")
    parser.add_argument("--peer-id", default=None, help="Other peer's id (default: opposite of --id)")
    parser.add_argument("--signaling-url", default="http://127.0.0.1:8000")
    parser.add_argument("--mode", choices=["stun", "preserve"], default="stun",
                         help="stun: discover public endpoint via STUN. preserve: assume port-preserving NAT")
    parser.add_argument("--port", type=int, default=random.randint(20000, 60000), help="Local UDP port (also used for STUN)")
    parser.add_argument("--stun-host", default="stun.l.google.com")
    parser.add_argument("--stun-port", type=int, default=19302)
    parser.add_argument("--quick", action="store_true", help="shrink benchmark sizes/durations for a fast smoke test")
    parser.add_argument("--experiment", choices=["default", "payload-sweep", "long-streaming", "long-duplex",
                                                  "loaded-latency", "multi-stream", "fast-path-compare"],
                         default="default", help="which validation experiment to run (see README)")
    parser.add_argument("--stream-duration", type=float, default=None,
                         help="override streaming duration in seconds (defaults depend on --experiment)")
    parser.add_argument("--pacing", choices=["max", "mbps", "pps"], default="max")
    parser.add_argument("--pacing-value", type=float, default=None, help="target rate for --pacing mbps/pps")
    parser.add_argument("--fast-path", action="store_true",
                         help="preallocated-buffer/raw-socket send path, to isolate Python overhead from network limits")
    args = parser.parse_args()
    if args.pacing != "max" and args.pacing_value is None:
        parser.error("--pacing-value is required when --pacing is 'mbps' or 'pps'")

    peer_id = args.peer_id or other_id(args.id)
    is_leader = args.id < peer_id
    idle_list = QUICK_IDLE_SECONDS if args.quick else DEFAULT_IDLE_SECONDS
    transfer_sizes = QUICK_TRANSFER_SIZES if args.quick else DEFAULT_TRANSFER_SIZES
    payload_sizes = QUICK_PAYLOAD_SIZES if args.quick else DEFAULT_PAYLOAD_SIZES
    probe_count = QUICK_PROBE_COUNT if args.quick else DEFAULT_PROBE_COUNT
    pacing_mbps = args.pacing_value if args.pacing == "mbps" else None
    pacing_pps = args.pacing_value if args.pacing == "pps" else None

    print(f"Peer {args.id} (mode={args.mode}, experiment={args.experiment}, leader={is_leader})")

    # The one and only UDP socket used for everything below.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    for opt in (socket.SO_RCVBUF, socket.SO_SNDBUF):
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.SOL_SOCKET, opt, SOCKET_BUFFER_REQUEST)
    sock.bind(("0.0.0.0", args.port))
    granted_rcvbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    granted_sndbuf = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    pacing_burst = max(16, min(4096, granted_rcvbuf // CHUNK_PAYLOAD // 4))
    print(f"Socket buffers: requested {SOCKET_BUFFER_REQUEST} bytes each; "
          f"kernel granted SO_RCVBUF={granted_rcvbuf}  SO_SNDBUF={granted_sndbuf}\n")

    if args.mode == "stun":
        own_ip, own_port = stun_get_mapped_address(sock, args.stun_host, args.stun_port)
    else:
        own_ip, own_port = None, args.port

    reg_resp = register(args.signaling_url, args.id, own_port)
    own_ip = own_ip or reg_resp["public_ip"]
    print("External endpoint:")
    print(f"{own_ip}:{own_port}\n")

    print(f"Waiting for peer '{peer_id}' and synchronized start time...")
    peer_info = wait_for_peer(args.signaling_url, args.id, peer_id)
    coordinator_peer_addr = (peer_info["public_ip"], peer_info["udp_port"])
    start_at = peer_info["start_at"]
    print("Peer endpoint:")
    print(f"{coordinator_peer_addr[0]}:{coordinator_peer_addr[1]}")

    delay = start_at - time.time()  # start_at is a wall-clock rendezvous shared across two machines - time.time() here is deliberate
    if delay > 0:
        print(f"Synchronized punch in {delay:.1f}s (assumes clocks are roughly in sync, e.g. via NTP)\n")
        await asyncio.sleep(delay)
    else:
        print("start_at already elapsed; punching immediately\n")

    loop = asyncio.get_running_loop()
    transport, protocol = await loop.create_datagram_endpoint(lambda: PeerProtocol(coordinator_peer_addr), sock=sock)
    protocol.pacing_burst = pacing_burst
    protocol.raw_sock = sock

    print("Punching...")
    punch_task = asyncio.create_task(punch_loop(protocol))
    for _ in range(50):
        if protocol.established:
            break
        await asyncio.sleep(0.1)
    punch_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await punch_task

    report: dict = {
        "self_id": args.id, "own_addr": (own_ip, own_port), "peer_addr": coordinator_peer_addr,
        "hole_punch_success": protocol.established,
        "buffers": {"requested": SOCKET_BUFFER_REQUEST, "rcvbuf": granted_rcvbuf, "sndbuf": granted_sndbuf},
    }

    if not protocol.established:
        print("Hole punch FAILED: no packet received from peer within timeout.")
        print_final_report(report)
        transport.close()
        return

    print("Connection established.")
    verified = protocol.peer_addr == coordinator_peer_addr
    report["direct_p2p_verified"] = verified
    if verified:
        print(f"Verified direct P2P: traffic origin {protocol.peer_addr} matches the coordinator-reported endpoint "
              f"(the coordinator has no data-plane endpoint - this traffic cannot be going through it).\n")
    else:
        print(f"NOTE: traffic origin {protocol.peer_addr} differs from the coordinator-reported "
              f"{coordinator_peer_addr} (NAT remapped before the first packet) - still direct P2P, just a "
              f"different mapping than initially advertised.\n")

    rebind_task = asyncio.create_task(rebind_watcher(protocol, REBIND_CHECK_INTERVAL))

    print(f"Running RTT / jitter / packet-loss benchmark ({probe_count} probes, idle baseline)...")
    probe_result = await run_probe_benchmark(protocol, probe_count, PROBE_INTERVAL)
    print(f"Idle RTT avg={probe_result['avg_rtt_ms']:.2f}ms p95={probe_result['p95_rtt_ms']:.2f}ms "
          f"p99={probe_result['p99_rtt_ms']:.2f}ms jitter={probe_result['jitter_ms']:.2f}ms "
          f"loss={probe_result['loss_pct']:.1f}%\n")
    report["probe"] = probe_result

    tid_counter = itertools.count(1)

    if args.experiment == "default":
        await run_default_experiment(protocol, report, tid_counter, transfer_sizes, payload_sizes,
                                      args, is_leader, pacing_mbps, pacing_pps)
    elif args.experiment == "payload-sweep":
        await run_payload_sweep_experiment(protocol, report, tid_counter, args)
    elif args.experiment == "long-streaming":
        await run_long_streaming_experiment(protocol, report, tid_counter, args, is_leader, pacing_mbps, pacing_pps)
    elif args.experiment == "long-duplex":
        await run_long_duplex_experiment(protocol, report, tid_counter, args, pacing_mbps, pacing_pps)
    elif args.experiment == "loaded-latency":
        await run_loaded_latency_experiment(protocol, report, tid_counter, args, pacing_mbps, pacing_pps)
    elif args.experiment == "multi-stream":
        await run_multi_stream_experiment(protocol, report, tid_counter, args, pacing_mbps, pacing_pps)
    elif args.experiment == "fast-path-compare":
        await run_fast_path_compare_experiment(protocol, report, tid_counter, args, pacing_mbps, pacing_pps)

    rebind_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await rebind_task
    report["rebind_events"] = protocol.rebind_event_count

    if args.experiment == "default":
        print("\nRunning NAT keepalive test...")
        report["keepalive"] = await run_keepalive_test(protocol, idle_list)

    print("\nWaiting for peer to finish its own benchmarks...")
    await barrier(protocol, b"ALL_DONE", timeout=180)

    print_final_report(report)
    transport.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
