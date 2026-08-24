# Profiling

`quic_dist`'s real, working diagnostic tool is `QUIC_DIST_RUST_DEBUG=1` —
used throughout this project's own debugging history (see
`docs/development-log.md`'s "Parallel-stream send/recv stall" and
"Real multi-micro-batch pipeline overlap" entries for two real bugs it
helped root-cause) but never previously documented for a repo consumer
to discover. This page documents what already exists; it doesn't add
new tooling.

## Enabling it

Set the env var on either rank (or both) before running any script that
uses `ProcessGroupQUIC`:

```bash
QUIC_DIST_RUST_DEBUG=1 python3 examples/bench_quic_dist.py
```

or for a real training run:

```bash
QUIC_DIST_RUST_DEBUG=1 python3 pipeline_finetune_rank.py configs/qwen25_0.5b_lora.yaml 0 <signaling_url>
```

This makes the Rust driver thread (`multiplexed_driver.rs`) print one
`[qdist-rs] ...` line to stderr per event of interest — most usefully, a
`TICK` line once per driver-loop iteration with a live snapshot of the
underlying QUIC connection's real state
(`engine.rs::debug_stats()`, reading `quinn_proto::Connection::stats()`
directly — not inferred, not estimated).

## Reading the output

Each `TICK` line looks like:

```
[qdist-rs] TICK poll_timeout=Some(45.2s) cwnd=212992 rtt=982.203µs congestion_events=0 lost_packets=0 lost_bytes=0 sent_packets=2063 udp_tx_datagrams=2061 udp_tx_bytes=2985216 udp_rx_datagrams=27 udp_rx_bytes=7675 frame_tx_stream=2055 frame_tx_acks=10 frame_rx_acks=24 frame_rx_stream_data_blocked=0 frame_rx_data_blocked=0 black_holes=0
```

| Field | Meaning |
|---|---|
| `poll_timeout` | Time until the driver loop's next scheduled wakeup (idle timeout, retransmission timer, etc.) — a very large value with no other activity means the connection is genuinely idle, not stuck. |
| `cwnd` | The congestion controller's current window, in bytes — how much may be in flight unacknowledged right now. Capped by `congestion.rs`'s `BoundedController` at the real OS socket buffer size (see `docs/architecture.md`). |
| `rtt` | Current round-trip-time estimate. Sub-millisecond means loopback or a very close real network path (see the parallel-stream-stall entry in `docs/development-log.md` for a real ~1ms cross-machine measurement). |
| `congestion_events` | Count of times the controller reacted to detected loss — nonzero means real packet loss occurred, not just slowness. |
| `lost_packets` / `lost_bytes` | quinn-proto's own loss-detection accounting. **Compare `lost_bytes` on the sender against `udp_tx_bytes - udp_rx_bytes` on the receiver** — if they match, it's real loss (not a false-positive heuristic); if they don't, something else is going on. |
| `sent_packets` | Cumulative packets sent on this connection. Frozen for many consecutive `TICK`s while data is still queued to send is the actual stall signature this tool was built to catch. |
| `udp_tx_bytes` / `udp_rx_bytes` | Cumulative bytes actually leaving/arriving at the UDP socket — the ground truth for "is data moving," independent of anything at the QUIC/application layer. |
| `frame_tx_*` / `frame_rx_*` | Counts of specific QUIC frame types sent/received (`stream`, `acks`, `stream_data_blocked`, `data_blocked`) — `stream_data_blocked`/`data_blocked` nonzero means the PEER told you it's flow-control-limited, a real, named reason for a stall rather than a guess. |
| `black_holes` | quinn-proto's own black-hole-path detection counter — should stay 0 on any working connection. |

## A worked example

The `TICK` line above is from a real, healthy transfer (see
`docs/development-log.md`): `cwnd=212992` sitting exactly at the real
kernel buffer ceiling, `lost_packets=0`/`lost_bytes=0`, `udp_tx_bytes`
climbing steadily across successive `TICK`s. A *stalled* connection
looks different in a specific, recognizable way: `sent_packets` frozen
across many consecutive ticks, `cwnd` also frozen (not growing, not
shrinking), and — critically — a real gap between what the sender's
`udp_tx_bytes` claims was sent and what the receiver's own `udp_rx_bytes`
shows arriving, matching `lost_bytes` exactly. That comparison (sender
stats vs. receiver stats, not just one side in isolation) is what
actually distinguishes "real network loss" from "the driver loop stopped
trying" — see `docs/development-log.md`'s full writeup for how this was
used to find and fix three independent, real bugs in one debugging pass.

## Known limitation

Python-level profiling (`py-spy` or similar) does **not** work in every
environment this project runs in — a real, hit limitation: `py-spy`
needs `ptrace`, which some containers' seccomp profile blocks outright
(see `examples/n3_cross_machine_rank.py`'s own comment on this). No
workaround is currently implemented; `QUIC_DIST_RUST_DEBUG=1` covers the
transport layer specifically and doesn't need `ptrace` at all (it's a
plain env-gated `eprintln!` inside the Rust driver thread itself).
