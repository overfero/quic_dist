// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `Engine`: drives one `quinn_proto::Endpoint` + (once established) one
//! `quinn_proto::Connection` for a single peer-to-peer link - the Rust
//! analogue of `quic_transport.py`'s `QUICTransport`/`_QuicAdapterProtocol`,
//! built on `quinn-proto` (sans-io) instead of `aioquic`.
//!
//! Unlike aioquic's single `QuicConnection` object, quinn-proto splits
//! per-socket state (`Endpoint`) from per-connection state (`Connection`),
//! with a bidirectional event-passing protocol between them
//! (`Connection::poll_endpoint_events()` -> `Endpoint::handle_event()` one
//! way, `Endpoint::handle()`'s `DatagramEvent::ConnectionEvent` ->
//! `Connection::handle_event()` the other way). This is unavoidable -
//! `Endpoint` is required even for exactly one connection, confirmed against
//! quinn-proto's own docs (there is no lighter-weight single-connection
//! constructor) - so `Engine` always owns both, draining the endpoint-event
//! loop to a fixed point after every operation that mutates the connection,
//! same as quinn's own internal driving pattern.
//!
//! No socket I/O of any kind happens in this crate (see module docstring
//! for `lib.rs`) - bytes come in via `receive_datagram`, bytes go out via
//! the `Vec<(Vec<u8>, SocketAddr)>` results returned from every method that
//! can produce them, mirroring `quic_transport.py`'s `_transmit()`.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Instant;

use bytes::{Bytes, BytesMut};
use quinn_proto::crypto::rustls::{QuicClientConfig, QuicServerConfig};
use quinn_proto::{
    ClientConfig, ConnectionHandle, DatagramEvent, Dir, Endpoint, EndpointConfig, Event,
    ServerConfig, StreamId, Transmit, TransportConfig, VarInt,
};

use crate::cert::{generate_self_signed_cert, SkipServerVerification};
use crate::congestion::BoundedControllerFactory;
use crate::error::{config, connect, protocol, stream, EngineError, Result};

/// One outgoing datagram, ready to hand to a real UDP socket on the Python
/// side (`sock.sendto(payload, addr)`).
/// One (possibly GSO-batched) outgoing transmission, ready to hand to a
/// real UDP socket. `segment_size` mirrors `quinn_proto::Transmit`'s own
/// field exactly: `None` means `data` is a single ordinary datagram; `Some(n)`
/// means `data` is actually several `n`-byte segments concatenated back to
/// back (the last one may be shorter) meant to be sent as ONE Linux GSO
/// batch (`sendmsg` with a `UDP_SEGMENT` control message) rather than one
/// `sendto()` per segment - see this crate's `poll_transmit` docstring for
/// why this matters (this is quinn-proto's OWN batching mechanism,
/// `enable_segmentation_offload: true` by default - it was simply never
/// being requested, see below).
pub struct OutDatagram {
    pub data: Vec<u8>,
    pub addr: SocketAddr,
    pub segment_size: Option<usize>,
}

/// A chunk of stream bytes plus its offset in the stream - see
/// `Engine::read_stream`'s docstring for why offset matters here (it
/// doesn't, for our single-ordered-stream design, but is kept for API
/// completeness/future use).
pub struct StreamChunk {
    pub offset: u64,
    pub bytes: Bytes,
}

/// Buffer-awareness knobs - see `TransportConfig.quic_max_data`'s comment in
/// `vllm/transport/base.py` for the full story: a window sized far beyond
/// what the real OS socket buffers can absorb caused a measured regression
/// for aioquic. The Python side reads the REAL granted `SO_RCVBUF`/
/// `SO_SNDBUF` via `getsockopt` after binding (mirroring
/// `udp_transport.py`'s own clamp) and passes the result in here - this
/// crate does not open or configure any socket itself, so it cannot read
/// these values on its own.
#[derive(Debug, Clone, Copy)]
pub struct WindowConfig {
    pub receive_window: u64,
    pub send_window: u64,
    pub stream_receive_window: u64,
    /// Hard ceiling on the congestion controller's reported window (see
    /// `congestion.rs`'s module docstring for why this exists: the default
    /// controller has no such ceiling on its own, and can grow past what
    /// the real OS socket buffers can hold for a large enough transfer,
    /// causing self-induced loss and a full stall). Should be derived from
    /// the same real, `getsockopt`-granted buffer size as the other window
    /// fields here, not set independently.
    pub max_congestion_window: u64,
}

pub struct EngineConfig {
    pub is_client: bool,
    pub idle_timeout_ms: u32,
    pub windows: WindowConfig,
}

/// Drives one QUIC connection to one peer. See module docstring.
pub struct Engine {
    endpoint: Endpoint,
    connection: Option<(ConnectionHandle, quinn_proto::Connection)>,
    is_client: bool,
    server_config: Option<Arc<ServerConfig>>,
    /// Scratch buffer reused across `poll_transmit`/`accept`/`handle` calls -
    /// quinn-proto's API writes packet bytes into a caller-provided
    /// `Vec<u8>` rather than allocating one per call.
    scratch: Vec<u8>,
}

fn build_transport_config(cfg: &EngineConfig) -> Result<Arc<TransportConfig>> {
    let mut transport = TransportConfig::default();
    transport
        .max_idle_timeout(Some(
            quinn_proto::IdleTimeout::try_from(std::time::Duration::from_millis(
                cfg.idle_timeout_ms as u64,
            ))
            .map_err(|error| config!("invalid idle timeout: {error}"))?,
        ))
        .receive_window(
            VarInt::from_u64(cfg.windows.receive_window)
                .map_err(|error| config!("receive_window out of range: {error}"))?,
        )
        .send_window(cfg.windows.send_window)
        .stream_receive_window(
            VarInt::from_u64(cfg.windows.stream_receive_window)
                .map_err(|error| config!("stream_receive_window out of range: {error}"))?,
        )
        .congestion_controller_factory(Arc::new(BoundedControllerFactory::new(
            Arc::new(quinn_proto::congestion::CubicConfig::default()),
            cfg.windows.max_congestion_window,
        )));
    Ok(Arc::new(transport))
}

fn crypto_provider() -> Arc<rustls::crypto::CryptoProvider> {
    Arc::new(rustls::crypto::ring::default_provider())
}

/// Same ALPN protocol string `quic_transport.py` uses (`ALPN_PROTOCOL =
/// "vllm-pp-v1"`) - QUIC's own TLS 1.3 profile (RFC 9001 §8.1) requires
/// ALPN to be present in the ClientHello and selected by the server, unlike
/// plain TLS where it's optional; omitting it risks a handshake failure,
/// not just a missing nicety.
const ALPN_PROTOCOL: &[u8] = b"vllm-pp-rs-v1";

impl Engine {
    /// Client role: generates its own connection ID and can send its first
    /// Initial packet immediately - mirrors `QUICTransport._connect_async`'s
    /// client branch (`quic.connect(...)` then `protocol._transmit()`).
    /// Returns the engine plus the Initial packet(s) to send right away.
    pub fn new_client(
        cfg: &EngineConfig,
        remote: SocketAddr,
        server_name: &str,
    ) -> Result<(Self, Vec<OutDatagram>)> {
        let transport = build_transport_config(cfg)?;
        let verifier = SkipServerVerification::new(crypto_provider());
        let mut rustls_config = rustls::ClientConfig::builder_with_provider(crypto_provider())
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(|error| config!("failed to select TLS 1.3: {error}"))?
            .dangerous()
            .with_custom_certificate_verifier(verifier)
            .with_no_client_auth();
        rustls_config.alpn_protocols = vec![ALPN_PROTOCOL.to_vec()];
        let quic_crypto = QuicClientConfig::try_from(rustls_config)
            .map_err(|error| config!("failed to build QUIC client crypto config: {error}"))?;
        let mut client_config = ClientConfig::new(Arc::new(quic_crypto));
        client_config.transport_config(transport);

        let endpoint_config = Arc::new(EndpointConfig::default());
        let mut endpoint = Endpoint::new(endpoint_config, None, true, None);

        let now = Instant::now();
        let (handle, connection) = endpoint
            .connect(now, client_config, remote, server_name)
            .map_err(|error| connect!("QUIC connect() failed: {error}"))?;

        let mut engine = Self {
            endpoint,
            connection: Some((handle, connection)),
            is_client: true,
            server_config: None,
            scratch: Vec::new(),
        };
        let initial = engine.poll_transmit(now);
        Ok((engine, initial))
    }

    /// Server role: the connection is constructed lazily, on the first
    /// Initial packet the peer sends - mirrors `_QuicAdapterProtocol.
    /// _handle_datagram`'s server branch (constructs aioquic's
    /// `QuicConnection` only once `original_destination_connection_id` is
    /// known from the client's own first packet). No address-spoofing
    /// protection via a Retry round-trip is used here either, for the same
    /// reason `quic_transport.py` skips it: the UDP hole punch already
    /// confirms both sides are a real, mutually-reachable pair before any
    /// QUIC traffic is sent at all.
    pub fn new_server(cfg: &EngineConfig) -> Result<Self> {
        let transport = build_transport_config(cfg)?;
        let (cert, key) = generate_self_signed_cert()?;
        // `ServerConfig::with_single_cert` (quinn-proto's own convenience
        // wrapper) would be simpler here, but it hides the underlying
        // rustls::ServerConfig, leaving no way to set `alpn_protocols` -
        // required by QUIC's TLS 1.3 profile (RFC 9001 §8.1), see
        // ALPN_PROTOCOL's own comment above - so this builds the
        // rustls::ServerConfig by hand instead, same shape as the client
        // side already needs for SkipServerVerification.
        let mut rustls_config = rustls::ServerConfig::builder_with_provider(crypto_provider())
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(|error| config!("failed to select TLS 1.3: {error}"))?
            .with_no_client_auth()
            .with_single_cert(vec![cert], key)
            .map_err(|error| config!("failed to install self-signed certificate: {error}"))?;
        rustls_config.alpn_protocols = vec![ALPN_PROTOCOL.to_vec()];
        let quic_crypto = QuicServerConfig::try_from(rustls_config)
            .map_err(|error| config!("failed to build QUIC server crypto config: {error}"))?;
        let mut server_config = ServerConfig::with_crypto(Arc::new(quic_crypto));
        server_config.transport_config(transport);
        let server_config = Arc::new(server_config);

        let endpoint_config = Arc::new(EndpointConfig::default());
        let endpoint = Endpoint::new(endpoint_config, Some(server_config.clone()), true, None);

        Ok(Self {
            endpoint,
            connection: None,
            is_client: false,
            server_config: Some(server_config),
            scratch: Vec::new(),
        })
    }

    /// Feeds one received UDP datagram in. `data` is the QUIC payload only
    /// (the caller - `vllm/transport/quic_rs_transport.py` - has already
    /// stripped the project's own `_QUIC_TAG` prefix and dispatched past
    /// the hole-punch protocol's own tags, exactly as
    /// `_QuicAdapterProtocol._handle_datagram` does for aioquic today).
    /// Returns any datagrams that must be sent back immediately as a direct
    /// result (e.g. version-negotiation/stateless-reset responses) - the
    /// caller should still call `poll_transmit` afterward for the general
    /// case, same as `_handle_datagram` -> `_transmit()` today.
    pub fn receive_datagram(
        &mut self,
        now: Instant,
        remote: SocketAddr,
        local_ip: Option<IpAddr>,
        data: BytesMut,
    ) -> Result<Vec<OutDatagram>> {
        self.scratch.clear();
        let event = self
            .endpoint
            .handle(now, remote, local_ip, None, data, &mut self.scratch);
        let Some(event) = event else {
            return Ok(Vec::new());
        };
        match event {
            DatagramEvent::ConnectionEvent(handle, connection_event) => {
                if let Some((known_handle, connection)) = self.connection.as_mut()
                    && *known_handle == handle {
                        connection.handle_event(connection_event);
                    }
                    // A handle mismatch here would mean a datagram routed to
                    // a connection this Engine never created - quinn-proto's
                    // own Endpoint bookkeeping is the source of truth for
                    // handle validity, so this is silently ignored rather
                    // than treated as an error: it cannot happen for a
                    // single-connection Engine unless quinn-proto itself has
                    // a bug, and erroring out on it would make an otherwise
                    // healthy connection fail for a condition this Engine
                    // has no way to independently verify.
                Ok(Vec::new())
            }
            DatagramEvent::NewConnection(incoming) => {
                if self.is_client || self.connection.is_some() {
                    // A second connection attempt arriving after we already
                    // have one (or arriving on a client-role engine, which
                    // never accepts) - never a fatal error for THIS engine,
                    // the peer will simply time out or retry.
                    return Ok(Vec::new());
                }
                self.scratch.clear();
                let (handle, connection) = self
                    .endpoint
                    .accept(
                        incoming,
                        now,
                        &mut self.scratch,
                        self.server_config.clone(),
                    )
                    .map_err(|error| protocol!("failed to accept new QUIC connection: {error:?}"))?;
                self.connection = Some((handle, connection));
                Ok(Vec::new())
            }
            DatagramEvent::Response(transmit) => Ok(vec![transmit_to_out(transmit, &self.scratch)]),
        }
    }

    /// Drains every pending outgoing datagram - mirrors `_transmit()`'s
    /// `datagrams_to_send()` loop. Always call this after `receive_datagram`/
    /// `handle_timeout`/any stream write, same as `quic_transport.py` does.
    ///
    /// Real bug found (not by guessing - by comparing against production
    /// QUIC implementation practice after this project's own loopback
    /// benchmark showed quic-rs trailing aioquic's pure-Python backend on
    /// raw throughput despite the state machine itself being in Rust):
    /// this used to call `connection.poll_transmit(now, 1, ...)` - the `1`
    /// is quinn-proto's `max_datagrams` parameter, "how many datagrams can
    /// be returned inside a single `Transmit` using GSO" (its own doc
    /// comment, `connection/mod.rs`). Passing `1` unconditionally defeats
    /// `enable_segmentation_offload` (on by default in `TransportConfig`)
    /// regardless of that setting - every outgoing packet was forced into
    /// its own separate `Transmit`, meaning one `sendto()` syscall (and one
    /// `_send_all` iteration, and one Vec<u8> allocation) per QUIC packet
    /// instead of batching many into one `sendmsg` with `UDP_SEGMENT`. This
    /// is exactly the kind of per-packet dispatch overhead this project's
    /// own profiling pointed at (~65-80% CPU during transfers, i.e.
    /// CPU-bound on something, not network-bound) - fixed by requesting a
    /// real batch size and threading `segment_size` through to the Python
    /// layer, which now does the actual GSO `sendmsg` (see
    /// `quic_rs_transport.py`'s `_send_all`). `MAX_GSO_DATAGRAMS = 10`
    /// matches the value quinn's own `quinn-udp` crate defaults to for the
    /// same purpose (not derived independently - there's no principled way
    /// to pick this without OS-level GSO-capability probing, which this
    /// crate doesn't do, so borrowing the ecosystem's own answer is more
    /// defensible than guessing a fresh number).
    /// One-line snapshot of quinn-proto's own real congestion/flow-control
    /// state - added specifically to investigate the parallel-stream
    /// cumulative-demand stall (see multiplexed_driver.rs's module
    /// docstring and the deliverable report's &sect;7): whether writes are
    /// genuinely blocked on congestion window, and what RTT/loss quinn-
    /// proto itself is observing, rather than guessing from the outside.
    pub fn debug_stats(&self) -> Option<String> {
        let (_, connection) = self.connection.as_ref()?;
        let stats = connection.stats();
        Some(format!(
            "cwnd={} rtt={:?} congestion_events={} lost_packets={} lost_bytes={} sent_packets={}",
            stats.path.cwnd,
            stats.path.rtt,
            stats.path.congestion_events,
            stats.path.lost_packets,
            stats.path.lost_bytes,
            stats.path.sent_packets,
        ))
    }

    pub fn poll_transmit(&mut self, now: Instant) -> Vec<OutDatagram> {
        self.drain_endpoint_events();
        const MAX_GSO_DATAGRAMS: usize = 10;
        let mut out = Vec::new();
        let Some((_, connection)) = self.connection.as_mut() else {
            return out;
        };
        loop {
            self.scratch.clear();
            match connection.poll_transmit(now, MAX_GSO_DATAGRAMS, &mut self.scratch) {
                Some(transmit) => out.push(transmit_to_out(transmit, &self.scratch)),
                None => break,
            }
        }
        out
    }

    /// Drains every pending application event - mirrors `_process_quic_
    /// events()`'s `next_event()` loop. `Event` is quinn-proto's own type;
    /// the PyO3 layer (Phase 4) is responsible for converting it to
    /// something Python can inspect (stream data ready / handshake done /
    /// connection lost), the same boundary `_on_stream_data`/handshake_
    /// waiter/`_on_terminated` sit at today for aioquic.
    pub fn poll_event(&mut self) -> Option<Event> {
        self.connection.as_mut().and_then(|(_, c)| c.poll())
    }

    /// Same as repeatedly calling `poll_event` until it returns `None`,
    /// batched into one call - added to cut down PyO3 round trips on the
    /// receive-side hot path (measured: `_process_events()`'s Python `while`
    /// loop calling `next_event()` once per event was a real, measurable
    /// contributor to this engine's per-datagram Python/FFI overhead, found
    /// while investigating why quic-rs's raw throughput trailed aioquic's
    /// `quic` backend on this project's own loopback benchmark despite
    /// moving the QUIC state machine itself to Rust). Behaviorally
    /// identical to the loop it replaces, just returned as one `Vec`
    /// instead of N separate calls.
    pub fn drain_events(&mut self) -> Vec<Event> {
        let Some((_, connection)) = self.connection.as_mut() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        while let Some(event) = connection.poll() {
            out.push(event);
        }
        out
    }

    pub fn poll_timeout(&mut self) -> Option<Instant> {
        self.connection.as_mut().and_then(|(_, c)| c.poll_timeout())
    }

    /// Fires a previously-scheduled timer - mirrors `_handle_timer()`.
    /// Caller must call `poll_transmit` afterward (same two-step shape as
    /// `_handle_timer` -> `_process_quic_events()` -> `_transmit()` today).
    pub fn handle_timeout(&mut self, now: Instant) {
        if let Some((_, connection)) = self.connection.as_mut() {
            connection.handle_timeout(now);
        }
    }

    /// Opens (idempotently is the CALLER's job, same as aioquic's
    /// `open_channel_stream`/`_out_stream_id` caching) one persistent
    /// unidirectional stream - see `quic_transport.py`'s module docstring
    /// for why exactly one stream per direction, reused for the whole
    /// connection lifetime, not one per message.
    pub fn open_uni_stream(&mut self) -> Result<StreamId> {
        let (_, connection) = self.connection.as_mut().ok_or(EngineError::NoConnection)?;
        connection
            .streams()
            .open(Dir::Uni)
            .ok_or_else(|| stream!("no unidirectional streams available (peer's limit exhausted)"))
    }

    /// Accepts a peer-opened unidirectional stream - call after observing
    /// `Event::Stream(StreamEvent::Opened { dir: Dir::Uni })` from
    /// `poll_event`. Returns `None` if there is currently no new incoming
    /// stream to accept (not an error - `Opened` can fire once for several
    /// streams becoming available at once, so callers should loop this
    /// until it returns `None`).
    pub fn accept_uni_stream(&mut self) -> Result<Option<StreamId>> {
        let (_, connection) = self.connection.as_mut().ok_or(EngineError::NoConnection)?;
        Ok(connection.streams().accept(Dir::Uni))
    }

    /// Writes to an already-open stream. Framing (the project's 8-byte
    /// length prefix) is the Python wrapper's job, not this engine's - see
    /// `WindowConfig`'s docstring analogue in `quic_transport.py`'s
    /// `_on_stream_data` for why framing stays out of the transport layer.
    ///
    /// Unlike aioquic's `send_stream_data` (which always accepts everything
    /// into an internal unbounded buffer, never fails or partially writes),
    /// quinn-proto's own `SendStream::write` is a real try-write: it can
    /// return fewer bytes than requested if only part of the flow-control
    /// window is currently available, and returns `WriteError::Blocked`
    /// (meaning "wrote zero, nothing available right now, retry after
    /// `StreamEvent::Writable`") rather than treating that as an error.
    /// Real bug hit building this: the first version propagated `Blocked`
    /// straight up to Python as a hard failure, and separately ignored the
    /// possibility of a partial write - together, under sustained streaming
    /// load (`test8_tensor_streaming.py`'s 1000-tensor run) this both
    /// crashed the caller on ordinary flow-control backpressure AND could
    /// have silently dropped bytes even when it didn't crash. `Blocked` is
    /// folded into the `Ok(0)` case here so the caller has one uniform
    /// signal ("wrote N of len(data) bytes, buffer the rest and retry
    /// later") instead of two different failure shapes to handle.
    pub fn write_stream(&mut self, id: StreamId, data: &[u8]) -> Result<usize> {
        let (_, connection) = self.connection.as_mut().ok_or(EngineError::NoConnection)?;
        match connection.send_stream(id).write(data) {
            Ok(n) => Ok(n),
            Err(quinn_proto::WriteError::Blocked) => Ok(0),
            Err(error) => Err(stream!("write to stream {id} failed: {error}")),
        }
    }

    /// Signals "no more data will ever be written to this stream" - the
    /// graceful-drain-before-close signal the TODO above (now resolved)
    /// asked for. `SendStream` itself exposes no direct "is every byte I
    /// wrote actually acknowledged" query (checked the compiled API
    /// directly, not just docs.rs summaries, per that TODO's own
    /// instruction) - but `finish()` plus waiting for the resulting
    /// `StreamEvent::Finished` (which quinn-proto's own docs say fires only
    /// once "a finished stream has been fully acknowledged or stopped")
    /// gives the same guarantee through the front door: the caller
    /// (`quic_rs_transport.py`'s `close()`) calls this once, then waits for
    /// a `"stream_finished"` event for this stream id before actually
    /// closing the connection, instead of closing immediately after the
    /// last `write_stream` call the way the first version of this engine
    /// did - confirmed by real testing (`test8_tensor_streaming.py`'s
    /// 1000-message run, and a smaller 200-message repro) to lose a
    /// meaningful tail of already-"successfully written" data otherwise,
    /// since a successful `write_stream` only means quinn-proto accepted
    /// the bytes into its OWN internal buffer, not that they were actually
    /// admitted past the congestion window onto the wire yet.
    pub fn finish_stream(&mut self, id: StreamId) -> Result<()> {
        let (_, connection) = self.connection.as_mut().ok_or(EngineError::NoConnection)?;
        connection
            .send_stream(id)
            .finish()
            .map_err(|error| stream!("finish stream {id} failed: {error}"))
    }

    /// Reads whatever is currently available on a stream, in order (`ordered
    /// = true` - the ordering guarantee this whole design depends on, same
    /// as aioquic's per-stream ordering that `_on_stream_data`'s docstring
    /// explains). Returns an empty Vec if nothing is available right now
    /// (not an error) - the caller polls again after the next `Event`.
    pub fn read_stream(&mut self, id: StreamId) -> Result<Vec<StreamChunk>> {
        let (_, connection) = self.connection.as_mut().ok_or(EngineError::NoConnection)?;
        let mut recv = connection.recv_stream(id);
        let mut chunks = match recv.read(true) {
            Ok(chunks) => chunks,
            Err(error) => return Err(stream!("read from stream {id} failed: {error}")),
        };
        let mut out = Vec::new();
        loop {
            match chunks.next(usize::MAX) {
                Ok(Some(chunk)) => out.push(StreamChunk {
                    offset: chunk.offset,
                    bytes: chunk.bytes,
                }),
                Ok(None) => break,
                // Not an error - "no more data available on this stream
                // RIGHT NOW", same as the stream simply having nothing new
                // (see `read_stream`'s own docstring: an empty Vec is the
                // normal "nothing yet" result, the caller polls again after
                // the next Event::Stream(StreamEvent::Readable)).
                Err(quinn_proto::ReadError::Blocked) => break,
                Err(error) => {
                    let _ = chunks.finalize();
                    return Err(stream!("read from stream {id} failed: {error}"));
                }
            }
        }
        let _ = chunks.finalize();
        Ok(out)
    }

    pub fn close(&mut self, now: Instant, error_code: u64, reason: &[u8]) {
        if let Some((_, connection)) = self.connection.as_mut() {
            let code = VarInt::from_u64(error_code).unwrap_or(VarInt::from_u32(0));
            connection.close(now, code, Bytes::copy_from_slice(reason));
        }
    }

    /// Drains the endpoint<->connection event exchange to a fixed point -
    /// must run after every `Connection` state mutation, same pattern the
    /// full `quinn` crate's own internal driver uses. See module docstring.
    fn drain_endpoint_events(&mut self) {
        let Some((handle, connection)) = self.connection.as_mut() else {
            return;
        };
        while let Some(endpoint_event) = connection.poll_endpoint_events() {
            if let Some(connection_event) = self.endpoint.handle_event(*handle, endpoint_event) {
                connection.handle_event(connection_event);
            }
        }
    }
}

fn transmit_to_out(transmit: Transmit, scratch: &[u8]) -> OutDatagram {
    OutDatagram {
        data: scratch[..transmit.size].to_vec(),
        addr: transmit.destination,
        segment_size: transmit.segment_size,
    }
}
