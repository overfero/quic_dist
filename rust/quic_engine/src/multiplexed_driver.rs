// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `MultiplexedConnectionDriver`: the multi-channel analogue of
//! `driver.rs`'s `ConnectionDriver` - one real QUIC connection, shared by
//! several independent named logical channels, each getting its own
//! persistent unidirectional stream pair (mirroring `driver.rs`'s single-
//! stream design, just keyed by channel name/stream id instead of having
//! exactly one of everything). This is the Rust-native replacement for
//! what `quic_broker.py` (aioquic) and `quic_rs_broker.py` (Python-
//! asyncio-orchestrated Rust engine) both drove via asyncio - see
//! `driver.rs`'s module docstring for why running the WHOLE connection on
//! a dedicated Rust thread (not just the protocol state machine) is the
//! point, and `project_quic_rs_rust_native_driver`/
//! `project_raw_udp_rs_production_readiness` in memory for the real bugs
//! found getting that single-channel design production-ready - every one
//! of those lessons (edge-triggered `stream_writable`, bounded-chunk
//! writes, batched event draining, GSO batching, drain-before-close via
//! `stream_finished`) applies per-channel here, not just once.
//!
//! Wire format matches `quic_broker.py`'s design exactly (so this stays a
//! pure implementation swap, not a protocol change): a new outbound
//! stream's first bytes are a self-describing preamble (2-byte big-endian
//! name length + UTF-8 channel name), followed by ordinary 8-byte-length-
//! prefixed messages exactly like `driver.rs`'s single-channel framing -
//! the receiving side learns "this raw QUIC stream_id belongs to channel
//! X" without any prior coordination about stream-id numbering.
//!
//! Same "no tag-byte prefix needed" simplification as `driver.rs`: this
//! socket is only ever handed over after hole-punch fully completes, so
//! nothing else ever reads it again.

use std::collections::{HashMap, HashSet};
use std::net::{SocketAddr, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bytes::BytesMut;
use quinn_proto::{Dir, Event, StreamEvent, StreamId};

use crate::driver::{send_gso, LEN_PREFIX_BYTES, MAX_POLL_INTERVAL, RECV_BUF_SIZE, WRITE_CHUNK_BYTES};
use crate::engine::{Engine, EngineConfig, OutDatagram};
use crate::error::{closed, connect, io, timeout, Result};

/// One-time preamble length prefix - see module docstring.
const CHANNEL_NAME_LEN_BYTES: usize = 2;

struct PendingSend {
    /// Full framed bytes: optional channel-name preamble (only present
    /// for a channel's very first send) + 8-byte length prefix + payload.
    data: Vec<u8>,
    offset: usize,
    done_tx: mpsc::Sender<Result<()>>,
}

struct OutboundChannel {
    stream_id: Option<StreamId>,
    pending: Option<PendingSend>,
    blocked_on_writable: bool,
}

impl Default for OutboundChannel {
    fn default() -> Self {
        Self { stream_id: None, pending: None, blocked_on_writable: false }
    }
}

struct InboundStream {
    channel: Option<String>,
    buf: Vec<u8>,
}

impl Default for InboundStream {
    fn default() -> Self {
        Self { channel: None, buf: Vec::new() }
    }
}

pub enum ChannelEvent {
    Message { channel: String, data: Vec<u8> },
    Closed(String),
}

struct QueuedSend {
    channel: String,
    payload: Vec<u8>,
    done_tx: mpsc::Sender<Result<()>>,
}

pub struct MultiplexedConnectionDriver {
    inbound_rx: std::sync::Mutex<mpsc::Receiver<ChannelEvent>>,
    outbound_tx: mpsc::Sender<QueuedSend>,
    shutdown: Arc<AtomicBool>,
    /// `Mutex`, not a plain field - see `driver.rs`'s `ConnectionDriver`'s
    /// identical field for the real PyO3-borrow-conflict bug this fixes
    /// (`close()` must be `&self`, callable concurrently with an
    /// in-flight `recv_any()` on another thread, which a `&mut self`
    /// `thread: Option<...>` field cannot support).
    thread: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
    /// See `driver.rs`'s `ConnectionDriver::close` docstring for the real
    /// wait-before-signal deadlock this replaced - identical fix here.
    drain_timeout_ms: Arc<std::sync::atomic::AtomicU64>,
}

struct ThreadState {
    engine: Engine,
    socket: UdpSocket,
    out_channels: HashMap<String, OutboundChannel>,
    in_streams: HashMap<StreamId, InboundStream>,
    max_message_bytes: usize,
    gso_supported: bool,
}

impl ThreadState {
    fn transmit(&mut self, now: Instant) {
        let out = self.engine.poll_transmit(now);
        self.send_all(out);
    }

    fn send_all(&mut self, datagrams: Vec<OutDatagram>) {
        for dg in datagrams {
            self.send_one(dg);
        }
    }

    fn send_one(&mut self, dg: OutDatagram) {
        let OutDatagram { data, addr, segment_size } = dg;
        let Some(segment_size) = segment_size else {
            let _ = self.socket.send_to(&data, addr);
            return;
        };
        if !self.gso_supported {
            self.send_segments_individually(&data, addr, segment_size);
            return;
        }
        if send_gso(&self.socket, &data, addr, segment_size as u16).is_err() {
            self.gso_supported = false;
            self.send_segments_individually(&data, addr, segment_size);
        }
    }

    fn send_segments_individually(&self, data: &[u8], addr: SocketAddr, segment_size: usize) {
        for offset in (0..data.len()).step_by(segment_size) {
            let end = (offset + segment_size).min(data.len());
            let _ = self.socket.send_to(&data[offset..end], addr);
        }
    }

    /// Reassembles complete length-prefixed messages from one raw QUIC
    /// stream, first consuming the one-time channel-name preamble if this
    /// is the stream's first data - see module docstring. Returns
    /// `Some((channel, message))` per complete message found (a single
    /// call can find more than one - handled by the caller looping).
    fn drain_stream_once(&mut self, stream_id: StreamId) -> Vec<(String, Vec<u8>)> {
        let mut out = Vec::new();
        let Ok(chunks) = self.engine.read_stream(stream_id) else {
            return out;
        };
        if chunks.is_empty() {
            return out;
        }
        let entry = self.in_streams.entry(stream_id).or_default();
        for chunk in chunks {
            entry.buf.extend_from_slice(&chunk.bytes);
        }
        loop {
            let entry = self.in_streams.get_mut(&stream_id).unwrap();
            if entry.channel.is_none() {
                if entry.buf.len() < CHANNEL_NAME_LEN_BYTES {
                    return out;
                }
                let name_len = u16::from_be_bytes([entry.buf[0], entry.buf[1]]) as usize;
                if entry.buf.len() < CHANNEL_NAME_LEN_BYTES + name_len {
                    return out;
                }
                let name_bytes = entry.buf[CHANNEL_NAME_LEN_BYTES..CHANNEL_NAME_LEN_BYTES + name_len].to_vec();
                entry.buf.drain(..CHANNEL_NAME_LEN_BYTES + name_len);
                entry.channel = Some(String::from_utf8_lossy(&name_bytes).into_owned());
                continue;
            }
            if entry.buf.len() < LEN_PREFIX_BYTES {
                return out;
            }
            let mut len_bytes = [0u8; 8];
            len_bytes.copy_from_slice(&entry.buf[..LEN_PREFIX_BYTES]);
            let msg_len = u64::from_be_bytes(len_bytes) as usize;
            if msg_len > self.max_message_bytes {
                // Caller (the driver loop) checks in_streams for this
                // condition separately via `find_oversized`, since we
                // can't cleanly signal "terminate the connection" from
                // inside this borrow - see the loop's own handling.
                return out;
            }
            let total_needed = LEN_PREFIX_BYTES + msg_len;
            if entry.buf.len() < total_needed {
                return out;
            }
            let message = entry.buf[LEN_PREFIX_BYTES..total_needed].to_vec();
            entry.buf.drain(..total_needed);
            let channel = entry.channel.clone().unwrap();
            out.push((channel, message));
        }
    }

    /// Attempts to make progress on one channel's in-flight send, if any -
    /// see `driver.rs`'s `drive_pending_send` for the identical edge-
    /// triggered-writable reasoning, just keyed by channel here.
    fn drive_channel_send(&mut self, channel_name: &str, now: Instant) {
        let Some(out_ch) = self.out_channels.get_mut(channel_name) else {
            return;
        };
        if out_ch.pending.is_none() {
            return;
        }
        if out_ch.blocked_on_writable {
            return; // still waiting on a fresh Writable edge for THIS channel's stream
        }
        if out_ch.stream_id.is_none() {
            match self.engine.open_uni_stream() {
                Ok(id) => out_ch.stream_id = Some(id),
                Err(error) => {
                    let pending = out_ch.pending.take().unwrap();
                    let _ = pending.done_tx.send(Err(error));
                    return;
                }
            }
        }
        let stream_id = out_ch.stream_id.unwrap();
        loop {
            let out_ch = self.out_channels.get_mut(channel_name).unwrap();
            let Some(pending) = out_ch.pending.as_mut() else { return };
            if pending.offset >= pending.data.len() {
                let pending = out_ch.pending.take().unwrap();
                let _ = pending.done_tx.send(Ok(()));
                return;
            }
            let end = (pending.offset + WRITE_CHUNK_BYTES).min(pending.data.len());
            let chunk_owned = pending.data[pending.offset..end].to_vec();
            match self.engine.write_stream(stream_id, &chunk_owned) {
                Ok(0) => {
                    let out_ch = self.out_channels.get_mut(channel_name).unwrap();
                    out_ch.blocked_on_writable = true;
                    self.transmit(now);
                    return;
                }
                Ok(n) => {
                    self.out_channels.get_mut(channel_name).unwrap().pending.as_mut().unwrap().offset += n;
                    self.transmit(now);
                    if n < chunk_owned.len() {
                        return; // partial, not blocked - retry next loop tick
                    }
                }
                Err(error) => {
                    let out_ch = self.out_channels.get_mut(channel_name).unwrap();
                    let pending = out_ch.pending.take().unwrap();
                    let _ = pending.done_tx.send(Err(error));
                    return;
                }
            }
        }
    }
}

impl MultiplexedConnectionDriver {
    pub fn connect_client(
        socket: UdpSocket,
        peer_addr: SocketAddr,
        server_name: &str,
        cfg: EngineConfig,
        max_message_bytes: usize,
        handshake_timeout: Duration,
    ) -> Result<Self> {
        let (engine, initial) = Engine::new_client(&cfg, peer_addr, server_name)?;
        Self::start(engine, socket, initial, max_message_bytes, handshake_timeout)
    }

    pub fn connect_server(
        socket: UdpSocket,
        cfg: EngineConfig,
        max_message_bytes: usize,
        handshake_timeout: Duration,
    ) -> Result<Self> {
        let engine = Engine::new_server(&cfg)?;
        Self::start(engine, socket, Vec::new(), max_message_bytes, handshake_timeout)
    }

    fn start(
        engine: Engine,
        socket: UdpSocket,
        initial: Vec<OutDatagram>,
        max_message_bytes: usize,
        handshake_timeout: Duration,
    ) -> Result<Self> {
        socket
            .set_read_timeout(Some(Duration::from_millis(1)))
            .map_err(|error| io!("failed to set socket read timeout: {error}"))?;

        let (inbound_tx, inbound_rx) = mpsc::channel();
        let (outbound_tx, outbound_rx) = mpsc::channel::<QueuedSend>();
        let (handshake_tx, handshake_rx) = mpsc::channel();
        let shutdown = Arc::new(AtomicBool::new(false));
        let drain_timeout_ms = Arc::new(std::sync::atomic::AtomicU64::new(200));

        let thread_shutdown = shutdown.clone();
        let thread_drained = Arc::new(AtomicBool::new(false)); // internal to the thread's own loop, not read outside it
        let thread_drain_timeout_ms = drain_timeout_ms.clone();
        let thread = std::thread::spawn(move || {
            run_driver_loop(
                ThreadState {
                    engine,
                    socket,
                    out_channels: HashMap::new(),
                    in_streams: HashMap::new(),
                    max_message_bytes,
                    gso_supported: true,
                },
                initial,
                inbound_tx,
                outbound_rx,
                handshake_tx,
                thread_shutdown,
                thread_drained,
                thread_drain_timeout_ms,
            );
        });

        match handshake_rx.recv_timeout(handshake_timeout) {
            Ok(Ok(())) => Ok(Self {
                inbound_rx: std::sync::Mutex::new(inbound_rx),
                outbound_tx,
                shutdown,
                thread: std::sync::Mutex::new(Some(thread)),
                drain_timeout_ms,
            }),
            Ok(Err(error)) => {
                shutdown.store(true, Ordering::SeqCst);
                let _ = thread.join();
                Err(error)
            }
            Err(_timeout) => {
                shutdown.store(true, Ordering::SeqCst);
                let _ = thread.join();
                Err(connect!("QUIC handshake did not complete within {}ms", handshake_timeout.as_millis()))
            }
        }
    }

    pub fn send_on_channel(&self, channel: String, data: Vec<u8>, timeout: Duration) -> Result<()> {
        let (done_tx, done_rx) = mpsc::channel();
        self.outbound_tx
            .send(QueuedSend { channel, payload: data, done_tx })
            .map_err(|_| closed!("connection driver thread is gone"))?;
        match done_rx.recv_timeout(timeout) {
            Ok(result) => result,
            Err(_) => Err(timeout!("send_on_channel() timed out after {}ms", timeout.as_millis())),
        }
    }

    /// Blocks for ONE channel event (a message on any channel, or the
    /// connection closing) - the caller is responsible for demultiplexing
    /// by the returned channel name (matches `quic_broker.py`'s own
    /// design: `_deliver`/`_channel_queues` did this same fan-out on the
    /// Python side; here the fan-out happens one level up, in the PyO3
    /// wrapper, so this stays a single ordered stream of events out of
    /// Rust - simpler and avoids needing N separate Rust-side queues for
    /// N channels that may not even be known about yet when the driver
    /// starts).
    pub fn recv_any(&self, timeout: Duration) -> Result<ChannelEvent> {
        let rx = self.inbound_rx.lock().unwrap();
        match rx.recv_timeout(timeout) {
            Ok(event) => Ok(event),
            Err(_) => Err(timeout!("recv_any() timed out after {}ms", timeout.as_millis())),
        }
    }

    /// See `driver.rs`'s `ConnectionDriver::close` docstring for the real
    /// wait-before-signal deadlock a prior version of this method had -
    /// identical fix here: set `shutdown` immediately, hand the real
    /// drain grace period to the thread via the shared atomic, then let
    /// `thread.join()` block for exactly as long as the thread's own
    /// shutdown handling takes. `&self`, not `&mut self` - see the
    /// `thread` field's own docstring for why (must be safely callable
    /// concurrently with an in-flight `recv_any()` on another thread).
    pub fn close(&self, drain_timeout: Duration) {
        self.drain_timeout_ms
            .store(drain_timeout.as_millis() as u64, Ordering::SeqCst);
        self.shutdown.store(true, Ordering::SeqCst);
        let handle = self.thread.lock().unwrap().take();
        if let Some(thread) = handle {
            let _ = thread.join();
        }
    }
}

fn run_driver_loop(
    mut state: ThreadState,
    initial: Vec<OutDatagram>,
    inbound_tx: mpsc::Sender<ChannelEvent>,
    outbound_rx: mpsc::Receiver<QueuedSend>,
    handshake_tx: mpsc::Sender<Result<()>>,
    shutdown: Arc<AtomicBool>,
    drained: Arc<AtomicBool>,
    drain_timeout_ms: Arc<std::sync::atomic::AtomicU64>,
) {
    if !initial.is_empty() {
        state.send_all(initial);
    }

    let mut handshake_tx = Some(handshake_tx);
    let mut recv_buf = vec![0u8; RECV_BUF_SIZE];
    let mut finish_requested = false;
    let mut shutdown_deadline: Option<Instant> = None;
    let mut pending_finish: HashSet<StreamId> = HashSet::new();

    loop {
        let now = Instant::now();

        let timer_deadline = state.engine.poll_timeout();
        let mut recv_timeout = MAX_POLL_INTERVAL;
        if let Some(deadline) = timer_deadline {
            recv_timeout = recv_timeout.min(deadline.saturating_duration_since(now).max(Duration::from_millis(1)));
        }
        let _ = state.socket.set_read_timeout(Some(recv_timeout));

        match state.socket.recv_from(&mut recv_buf) {
            Ok((n, from)) => {
                let data = BytesMut::from(&recv_buf[..n]);
                if let Ok(responses) = state.engine.receive_datagram(Instant::now(), from, None, data) {
                    state.send_all(responses);
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock || error.kind() == std::io::ErrorKind::TimedOut => {}
            Err(error) => {
                finish_all(&inbound_tx, &mut handshake_tx, format!("socket recv error: {error}"));
                break;
            }
        }

        let now = Instant::now();
        if let Some(deadline) = state.engine.poll_timeout() {
            if now >= deadline {
                state.engine.handle_timeout(now);
            }
        }

        let mut readable_streams: HashSet<StreamId> = HashSet::new();
        let mut writable_channels: HashSet<String> = HashSet::new();
        let mut terminated: Option<String> = None;
        for event in state.engine.drain_events() {
            match event {
                Event::Connected => {
                    if let Some(tx) = handshake_tx.take() {
                        let _ = tx.send(Ok(()));
                    }
                }
                Event::ConnectionLost { reason } => {
                    terminated = Some(format!("{reason:?}"));
                }
                Event::Stream(StreamEvent::Opened { dir: Dir::Uni }) => {
                    while let Ok(Some(id)) = state.engine.accept_uni_stream() {
                        state.in_streams.entry(id).or_default();
                        readable_streams.insert(id);
                    }
                }
                Event::Stream(StreamEvent::Readable { id }) => {
                    readable_streams.insert(id);
                }
                Event::Stream(StreamEvent::Writable { id }) => {
                    // Find which channel owns this stream id and clear
                    // its block - a linear scan is fine here (channel
                    // counts are small, single digits in this project's
                    // real topology, not a hot per-message path).
                    for (name, out_ch) in state.out_channels.iter_mut() {
                        if out_ch.stream_id == Some(id) {
                            out_ch.blocked_on_writable = false;
                            writable_channels.insert(name.clone());
                            break;
                        }
                    }
                }
                Event::Stream(StreamEvent::Finished { id }) => {
                    pending_finish.remove(&id);
                }
                _ => {}
            }
        }
        if let Some(reason) = terminated {
            finish_all(&inbound_tx, &mut handshake_tx, reason);
            break;
        }

        let mut oversized: Option<String> = None;
        for stream_id in readable_streams {
            let messages = state.drain_stream_once(stream_id);
            for (channel, data) in messages {
                let _ = inbound_tx.send(ChannelEvent::Message { channel, data });
            }
            if let Some(entry) = state.in_streams.get(&stream_id) {
                let len_known = entry.buf.len() >= LEN_PREFIX_BYTES && entry.channel.is_some();
                if len_known {
                    let mut len_bytes = [0u8; 8];
                    len_bytes.copy_from_slice(&entry.buf[..LEN_PREFIX_BYTES]);
                    let msg_len = u64::from_be_bytes(len_bytes) as usize;
                    if msg_len > state.max_message_bytes {
                        oversized = Some(format!(
                            "message on stream {stream_id} (channel {:?}) exceeded {} bytes",
                            entry.channel, state.max_message_bytes
                        ));
                    }
                }
            }
        }
        if let Some(reason) = oversized {
            state.engine.close(Instant::now(), 1, b"message too large");
            state.transmit(Instant::now());
            finish_all(&inbound_tx, &mut handshake_tx, reason);
            break;
        }

        // Make progress on any channel that just got a fresh Writable
        // edge, AND on any channel that has a brand-new pending send
        // waiting for its stream to even be opened yet (never blocked,
        // so not gated on the edge above).
        for name in writable_channels {
            state.drive_channel_send(&name, Instant::now());
        }

        if let Ok(queued) = outbound_rx.try_recv() {
            let out_ch = state.out_channels.entry(queued.channel.clone()).or_default();
            if out_ch.pending.is_none() {
                let mut framed = Vec::with_capacity(LEN_PREFIX_BYTES + queued.payload.len() + 16);
                if out_ch.stream_id.is_none() {
                    let name_bytes = queued.channel.as_bytes();
                    framed.extend_from_slice(&(name_bytes.len() as u16).to_be_bytes());
                    framed.extend_from_slice(name_bytes);
                }
                framed.extend_from_slice(&(queued.payload.len() as u64).to_be_bytes());
                framed.extend_from_slice(&queued.payload);
                out_ch.pending = Some(PendingSend { data: framed, offset: 0, done_tx: queued.done_tx });
                state.drive_channel_send(&queued.channel, Instant::now());
            } else {
                // Should not happen given callers await completion before
                // sending the next message on the SAME channel (matches
                // `quic_broker.py`'s own one-in-flight-per-channel
                // contract) - fail loudly rather than silently drop.
                let _ = queued.done_tx.send(Err(closed!(
                    "channel {:?} already has an in-flight send",
                    queued.channel
                )));
            }
        }

        state.transmit(Instant::now());

        if shutdown.load(Ordering::SeqCst) {
            if !finish_requested {
                finish_requested = true;
                let grace_ms = drain_timeout_ms.load(Ordering::SeqCst);
                shutdown_deadline = Some(Instant::now() + Duration::from_millis(grace_ms));
                for out_ch in state.out_channels.values() {
                    if let Some(id) = out_ch.stream_id {
                        if state.engine.finish_stream(id).is_ok() {
                            pending_finish.insert(id);
                        }
                    }
                }
                state.transmit(Instant::now());
                if pending_finish.is_empty() {
                    drained.store(true, Ordering::SeqCst);
                }
            }
            if pending_finish.is_empty() {
                drained.store(true, Ordering::SeqCst);
            }
            let past_deadline = shutdown_deadline.is_some_and(|deadline| Instant::now() > deadline);
            if drained.load(Ordering::SeqCst) || past_deadline {
                state.engine.close(Instant::now(), 0, b"");
                state.transmit(Instant::now());
                break;
            }
        }
    }
}

fn finish_all(inbound_tx: &mpsc::Sender<ChannelEvent>, handshake_tx: &mut Option<mpsc::Sender<Result<()>>>, reason: String) {
    if let Some(tx) = handshake_tx.take() {
        let _ = tx.send(Err(connect!("{reason}")));
    }
    let _ = inbound_tx.send(ChannelEvent::Closed(reason));
}
