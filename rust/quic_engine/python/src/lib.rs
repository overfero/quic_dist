// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Thin PyO3 bindings for `vllm_quic_engine::Engine`.
//!
//! Phase 4 of `/root/.claude/plans/frolicking-swimming-panda.md`. Mirrors
//! exactly what `_QuicAdapterProtocol`/`QUICTransport` already call on
//! aioquic's `QuicConnection` today (`receive_datagram`, `poll_transmit`,
//! `next_event`, `get_timer`/`handle_timer`, stream open/write/read,
//! `close`) - see `quic_transport.py` for the Python-side integration shape
//! this is meant to slot into unchanged in Phase 5.
//!
//! `std::time::Instant` never crosses this boundary - every method computes
//! `Instant::now()` internally (same as `quic_transport.py` computing
//! `now = self.loop.time()` fresh at each call site, not threading a clock
//! value through Python). `poll_timeout` instead returns a relative
//! duration in seconds, since that is what `asyncio.get_event_loop().call_later()`
//! actually wants on the Python side.
//!
//! Every `#[pymethods]` entry point returns `PyResult<_>`, never
//! `.unwrap()`/`panic!` on network-controlled input - required because this
//! workspace builds with `panic = "abort"` (see `error.rs`'s module
//! docstring): a Rust panic crossing into Python would abort the whole
//! Python process, not raise a catchable exception.

use std::net::{IpAddr, SocketAddr};
use std::time::{Duration, Instant};

use bytes::BytesMut;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vllm_quic_engine::{Dir, Engine, EngineConfig, Event, StreamEvent, StreamId, VarInt, WindowConfig};

fn parse_stream_id(raw: u64) -> PyResult<StreamId> {
    VarInt::from_u64(raw)
        .map(StreamId::from)
        .map_err(|error| PyValueError::new_err(format!("invalid stream id {raw}: {error}")))
}

fn to_py_err<E: std::fmt::Display>(error: E) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn parse_addr(host: &str, port: u16) -> PyResult<SocketAddr> {
    let ip: IpAddr = host
        .parse()
        .map_err(|error| PyValueError::new_err(format!("invalid IP address {host:?}: {error}")))?;
    Ok(SocketAddr::new(ip, port))
}

fn addr_to_py(addr: SocketAddr) -> (String, u16) {
    (addr.ip().to_string(), addr.port())
}

/// `(data, (host, port), segment_size)` - `segment_size` is `None` for an
/// ordinary single datagram, or `Some(n)` when `data` is actually several
/// `n`-byte segments meant to be sent as ONE GSO batch - see
/// `quic_rs_transport.py`'s `_send_all` for how the Python side acts on
/// this (a real `sendmsg`/`UDP_SEGMENT` GSO send, not a fake/ignored field).
type PyDatagram = (Vec<u8>, (String, u16), Option<usize>);

fn datagrams_to_py(datagrams: Vec<vllm_quic_engine::OutDatagram>) -> Vec<PyDatagram> {
    datagrams
        .into_iter()
        .map(|d| (d.data, addr_to_py(d.addr), d.segment_size))
        .collect()
}

/// One application event, as `(kind, stream_id, detail)`:
/// - `("connected", None, None)`
/// - `("connection_lost", None, Some(reason))`
/// - `("stream_readable", Some(id), None)`
/// - `("stream_writable", Some(id), None)` - a stream that previously
///   returned 0 from `write_stream` (flow-control blocked) might accept
///   more now - retry the write.
/// - `("stream_opened", None, None)` - caller should loop `accept_uni_stream()`
/// - `("stream_available", None, None)` - surfaced for completeness (at
///   least one new outbound stream slot may be opened); this design opens
///   exactly one outbound stream per connection and never needs more, so
///   this should never actually drive any caller behavior.
/// - `("stream_finished", Some(id), None)`
/// - `("stream_stopped", Some(id), Some(error_code_as_str))`
/// - `("datagram_received", None, None)` / `("datagrams_unblocked", None, None)`
///   - surfaced for completeness; this design never sends unreliable
///     datagrams, so these should never actually occur in practice.
type PyEvent = (String, Option<u64>, Option<String>);

fn event_to_py(event: Event) -> PyEvent {
    match event {
        Event::HandshakeDataReady => ("handshake_data_ready".to_string(), None, None),
        Event::Connected => ("connected".to_string(), None, None),
        Event::ConnectionLost { reason } => {
            ("connection_lost".to_string(), None, Some(reason.to_string()))
        }
        Event::Stream(StreamEvent::Readable { id }) => {
            ("stream_readable".to_string(), Some(u64::from(id)), None)
        }
        Event::Stream(StreamEvent::Writable { id }) => {
            ("stream_writable".to_string(), Some(u64::from(id)), None)
        }
        Event::Stream(StreamEvent::Available { dir: _ }) => {
            ("stream_available".to_string(), None, None)
        }
        Event::Stream(StreamEvent::Opened { dir: Dir::Uni }) => {
            ("stream_opened".to_string(), None, None)
        }
        Event::Stream(StreamEvent::Opened { dir: Dir::Bi }) => {
            // This design never opens bidirectional streams - see
            // quic_transport.py's module docstring for why (one persistent
            // unidirectional stream per direction, not bidirectional). A
            // peer that somehow opened one anyway is surfaced generically
            // rather than silently dropped.
            ("stream_opened_bidi_unexpected".to_string(), None, None)
        }
        Event::Stream(StreamEvent::Finished { id }) => {
            ("stream_finished".to_string(), Some(u64::from(id)), None)
        }
        Event::Stream(StreamEvent::Stopped { id, error_code }) => (
            "stream_stopped".to_string(),
            Some(u64::from(id)),
            Some(error_code.to_string()),
        ),
        Event::DatagramReceived => ("datagram_received".to_string(), None, None),
        Event::DatagramsUnblocked => ("datagrams_unblocked".to_string(), None, None),
    }
}

#[allow(clippy::too_many_arguments)]
fn engine_config(
    idle_timeout_ms: u32,
    receive_window: u64,
    send_window: u64,
    stream_receive_window: u64,
    max_congestion_window: u64,
    is_client: bool,
) -> EngineConfig {
    EngineConfig {
        is_client,
        idle_timeout_ms,
        windows: WindowConfig {
            receive_window,
            send_window,
            stream_receive_window,
            max_congestion_window,
        },
    }
}

/// Thin PyO3 wrapper around a single `Engine` (no dedicated background
/// thread, no socket ownership of its own - the caller drives I/O).
/// **Intentionally unused by `ProcessGroupQUIC`** in `quic_dist` - see
/// `PyMultiplexedConnectionDriver`'s docstring below for what's actually
/// used, and `driver.rs`'s module docstring for why this lower-level
/// class is kept around anyway (a reference implementation, not dead
/// code left over from a migration).
#[pyclass]
struct PyQuicEngine {
    inner: Engine,
}

#[pymethods]
impl PyQuicEngine {
    /// Client role - see `Engine::new_client`. Returns `(engine,
    /// initial_datagrams)`; the caller must send `initial_datagrams` on its
    /// UDP socket right away (mirrors `QUICTransport._connect_async`'s
    /// `quic.connect(...)` then `protocol._transmit()`).
    #[staticmethod]
    #[pyo3(signature = (remote_host, remote_port, server_name, idle_timeout_ms, receive_window, send_window, stream_receive_window, max_congestion_window))]
    #[allow(clippy::too_many_arguments)]
    fn new_client(
        remote_host: &str,
        remote_port: u16,
        server_name: &str,
        idle_timeout_ms: u32,
        receive_window: u64,
        send_window: u64,
        stream_receive_window: u64,
        max_congestion_window: u64,
    ) -> PyResult<(Self, Vec<PyDatagram>)> {
        let remote = parse_addr(remote_host, remote_port)?;
        let cfg = engine_config(
            idle_timeout_ms, receive_window, send_window, stream_receive_window,
            max_congestion_window, true,
        );
        let (engine, initial) =
            Engine::new_client(&cfg, remote, server_name).map_err(to_py_err)?;
        Ok((Self { inner: engine }, datagrams_to_py(initial)))
    }

    /// Server role - see `Engine::new_server`. The connection is
    /// constructed lazily on the first datagram received.
    #[staticmethod]
    #[pyo3(signature = (idle_timeout_ms, receive_window, send_window, stream_receive_window, max_congestion_window))]
    fn new_server(
        idle_timeout_ms: u32,
        receive_window: u64,
        send_window: u64,
        stream_receive_window: u64,
        max_congestion_window: u64,
    ) -> PyResult<Self> {
        let cfg = engine_config(
            idle_timeout_ms, receive_window, send_window, stream_receive_window,
            max_congestion_window, false,
        );
        let engine = Engine::new_server(&cfg).map_err(to_py_err)?;
        Ok(Self { inner: engine })
    }

    /// Feeds one received UDP datagram in - `data` is the QUIC payload only
    /// (caller has already stripped this project's `_QUIC_TAG` prefix and
    /// dispatched past the hole-punch protocol's own tags). Returns any
    /// datagrams to send back immediately; caller should still call
    /// `poll_transmit` afterward for the general case.
    fn receive_datagram(
        &mut self,
        data: &[u8],
        remote_host: &str,
        remote_port: u16,
    ) -> PyResult<Vec<PyDatagram>> {
        let remote = parse_addr(remote_host, remote_port)?;
        let out = self
            .inner
            .receive_datagram(Instant::now(), remote, None, BytesMut::from(data))
            .map_err(to_py_err)?;
        Ok(datagrams_to_py(out))
    }

    /// Drains every pending outgoing datagram - call after
    /// `receive_datagram`/`handle_timeout`/any stream write.
    fn poll_transmit(&mut self) -> Vec<PyDatagram> {
        datagrams_to_py(self.inner.poll_transmit(Instant::now()))
    }

    /// Pops the next pending application event, or `None` if there is
    /// nothing new right now. See `PyEvent`'s doc comment for the shape.
    fn next_event(&mut self) -> Option<PyEvent> {
        self.inner.poll_event().map(event_to_py)
    }

    /// Same events as repeated `next_event()` calls, batched into one PyO3
    /// round trip - see `Engine::drain_events`'s docstring for why. Prefer
    /// this over `next_event()` in a loop on any hot path.
    fn drain_events(&mut self) -> Vec<PyEvent> {
        self.inner.drain_events().into_iter().map(event_to_py).collect()
    }

    /// Seconds from now until the next timer should fire, or `None` if no
    /// timer is currently scheduled - feed directly to
    /// `asyncio.get_event_loop().call_later(seconds, ...)`.
    fn poll_timeout_seconds(&mut self) -> Option<f64> {
        self.inner
            .poll_timeout()
            .map(|deadline| deadline.saturating_duration_since(Instant::now()).as_secs_f64())
    }

    /// Fires a previously-scheduled timer. Caller must call `poll_transmit`
    /// afterward (same two-step shape `_handle_timer` uses today).
    fn handle_timeout(&mut self) {
        self.inner.handle_timeout(Instant::now());
    }

    /// Opens one persistent unidirectional stream - idempotency (caching
    /// the returned id so this is only called once per connection
    /// lifetime) is the CALLER's job, same as aioquic's
    /// `_out_stream_id` caching in `quic_transport.py`.
    fn open_uni_stream(&mut self) -> PyResult<u64> {
        self.inner.open_uni_stream().map(u64::from).map_err(to_py_err)
    }

    /// Accepts one peer-opened unidirectional stream - call after
    /// observing a `"stream_opened"` event, in a loop until this returns
    /// `None` (a single event can correspond to several newly-available
    /// streams).
    fn accept_uni_stream(&mut self) -> PyResult<Option<u64>> {
        self.inner
            .accept_uni_stream()
            .map(|maybe_id| maybe_id.map(u64::from))
            .map_err(to_py_err)
    }

    /// Writes to an already-open stream. Framing (this project's 8-byte
    /// length prefix) is the Python wrapper's job, not this engine's.
    ///
    /// `data` should be a bounded-size chunk, not "everything left to
    /// send" - see `send_message`'s docstring in quic_rs_transport.py for
    /// why: passing the full remaining tail on every retry iteration
    /// forces a copy of that whole (shrinking, but still often large) tail
    /// somewhere on the way into this call regardless of which side does
    /// it, which is effectively quadratic in message size for a large
    /// message needing many iterations. Confirmed directly:
    /// `test22_tensor_500mb_memory.py`'s 500MB message never completed
    /// within any reasonable timeout passing the full remaining tail each
    /// time; capping the Python caller's chunk size fixed it without
    /// needing to change this signature at all.
    fn write_stream(&mut self, stream_id: u64, data: &[u8]) -> PyResult<usize> {
        self.inner
            .write_stream(parse_stream_id(stream_id)?, data)
            .map_err(to_py_err)
    }

    /// Reads whatever is currently available on a stream, in order, and
    /// returns it concatenated as one `bytes` object (chunk boundaries
    /// don't matter for this project's length-prefix framing, which
    /// re-buffers on the Python side regardless). Returns an empty `bytes`
    /// if nothing is available right now (not an error) - the caller polls
    /// again after the next `"stream_readable"` event.
    fn read_stream<'py>(&mut self, py: Python<'py>, stream_id: u64) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let chunks = self.inner.read_stream(parse_stream_id(stream_id)?).map_err(to_py_err)?;
        let total: usize = chunks.iter().map(|c| c.bytes.len()).sum();
        let mut buf = Vec::with_capacity(total);
        for chunk in chunks {
            buf.extend_from_slice(&chunk.bytes);
        }
        Ok(pyo3::types::PyBytes::new(py, &buf))
    }

    /// Signals "no more data will ever be written to this stream" and
    /// waits (via `"stream_finished"`) for confirmation that everything
    /// already written was actually acknowledged - see `Engine::
    /// finish_stream`'s docstring for why this exists and what real bug it
    /// fixes. Call this before `close()` for every stream this side ever
    /// wrote to.
    fn finish_stream(&mut self, stream_id: u64) -> PyResult<()> {
        self.inner
            .finish_stream(parse_stream_id(stream_id)?)
            .map_err(to_py_err)
    }

    fn close(&mut self, error_code: u64, reason: &[u8]) {
        self.inner.close(Instant::now(), error_code, reason);
    }
}

/// PyO3 binding for `vllm_quic_engine::ConnectionDriver` - unlike
/// `PyQuicEngine` above (sans-io, every method a single small step Python
/// drives one at a time via asyncio), this one owns a real socket and
/// drives the ENTIRE connection lifetime - handshake, timers, GSO send,
/// stream framing, drain-before-close - on a dedicated Rust thread. See
/// `driver.rs`'s module docstring for the full rationale and exactly
/// which `quic_rs_transport.py` bug fixes this ports. **Intentionally
/// unused by `ProcessGroupQUIC`** in `quic_dist` - see
/// `PyMultiplexedConnectionDriver` below for what's actually used;
/// `driver.rs`'s own docstring explains why this single-channel driver
/// is kept anyway (a reference implementation used to diff real bugs
/// against, not dead code).
#[pyclass]
struct PyQuicConnectionDriver {
    inner: vllm_quic_engine::ConnectionDriver,
}

#[pymethods]
impl PyQuicConnectionDriver {
    /// `fd` is the Python socket's `sock.fileno()`, already hole-punched
    /// and otherwise idle (no other protocol reads/writes it again after
    /// this call) - see `driver.rs`'s module docstring. Duplicates the fd
    /// (same rationale as `udp_raw_engine`'s `PyRawUdpEngine::from_fd`:
    /// the Python `socket.socket` object keeps independently owning and
    /// eventually closing its own copy). Blocks (GIL released) until the
    /// QUIC handshake completes or `handshake_timeout_ms` elapses.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        fd, remote_host, remote_port, server_name, idle_timeout_ms,
        receive_window, send_window, stream_receive_window, max_congestion_window,
        max_message_bytes, handshake_timeout_ms,
    ))]
    fn connect_client(
        py: Python<'_>,
        fd: i32,
        remote_host: &str,
        remote_port: u16,
        server_name: &str,
        idle_timeout_ms: u32,
        receive_window: u64,
        send_window: u64,
        stream_receive_window: u64,
        max_congestion_window: u64,
        max_message_bytes: usize,
        handshake_timeout_ms: u64,
    ) -> PyResult<Self> {
        let socket = dup_socket(fd)?;
        let remote = parse_addr(remote_host, remote_port)?;
        let cfg = engine_config(
            idle_timeout_ms, receive_window, send_window, stream_receive_window,
            max_congestion_window, true,
        );
        let inner = py
            .detach(|| {
                vllm_quic_engine::ConnectionDriver::connect_client(
                    socket, remote, server_name, cfg, max_message_bytes,
                    Duration::from_millis(handshake_timeout_ms),
                )
            })
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Server role - see `ConnectionDriver::connect_server`. Blocks (GIL
    /// released) until a client's handshake completes or
    /// `handshake_timeout_ms` elapses.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        fd, idle_timeout_ms, receive_window, send_window, stream_receive_window,
        max_congestion_window, max_message_bytes, handshake_timeout_ms,
    ))]
    fn connect_server(
        py: Python<'_>,
        fd: i32,
        idle_timeout_ms: u32,
        receive_window: u64,
        send_window: u64,
        stream_receive_window: u64,
        max_congestion_window: u64,
        max_message_bytes: usize,
        handshake_timeout_ms: u64,
    ) -> PyResult<Self> {
        let socket = dup_socket(fd)?;
        let cfg = engine_config(
            idle_timeout_ms, receive_window, send_window, stream_receive_window,
            max_congestion_window, false,
        );
        let inner = py
            .detach(|| {
                vllm_quic_engine::ConnectionDriver::connect_server(
                    socket, cfg, max_message_bytes, Duration::from_millis(handshake_timeout_ms),
                )
            })
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Blocks (GIL released) until `data` has been fully handed to the
    /// stream's send buffer, or `timeout_ms` elapses, or the connection
    /// is lost - see `ConnectionDriver::send`'s docstring.
    fn send(&self, py: Python<'_>, data: Vec<u8>, timeout_ms: u64) -> PyResult<()> {
        py.detach(|| self.inner.send(data, Duration::from_millis(timeout_ms)))
            .map_err(to_py_err)
    }

    /// Blocks (GIL released) until one complete message arrives, or
    /// `timeout_ms` elapses, or the connection is lost.
    fn recv<'py>(&self, py: Python<'py>, timeout_ms: u64) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
        let data = py
            .detach(|| self.inner.recv(Duration::from_millis(timeout_ms)))
            .map_err(to_py_err)?;
        Ok(pyo3::types::PyBytes::new(py, &data))
    }

    /// Graceful drain-before-close, then a real shutdown - see
    /// `ConnectionDriver::close`'s docstring. Blocks (GIL released) up to
    /// `drain_timeout_ms` waiting for already-written data to actually
    /// be acknowledged before tearing the connection down.
    ///
    /// `&self`, not `&mut self` - real bug found and fixed here: PyO3
    /// enforces Rust's borrow rules dynamically on `#[pyclass]` objects,
    /// so a `&mut self` method here could never actually run while
    /// another thread's `recv()` call (which holds a shared borrow for
    /// its ENTIRE blocking duration, even with `py.detach()` releasing
    /// the GIL - that only releases the GIL, not PyO3's own borrow
    /// guard) was still in flight. Confirmed directly: a background
    /// thread blocked in `recv()` while `close()` was called from
    /// another raised `RuntimeError: Already borrowed` from Python - and
    /// since that happens BEFORE this method body ever runs, `shutdown`
    /// was never actually set, hanging the driver thread (and the whole
    /// process) forever. `ConnectionDriver::close` itself was already
    /// `&self`-safe (interior mutability throughout) before this fix;
    /// only this wrapper's signature was the problem.
    fn close(&self, py: Python<'_>, drain_timeout_ms: u64) {
        py.detach(|| self.inner.close(Duration::from_millis(drain_timeout_ms)));
    }
}

/// PyO3 binding for `vllm_quic_engine::MultiplexedConnectionDriver` - the
/// multi-channel analogue of `PyQuicConnectionDriver`, backing this
/// project's `"quic-shared"` broker (several named logical channels
/// sharing one real QUIC connection). See `multiplexed_driver.rs`'s
/// module docstring for the full design.
#[pyclass]
struct PyMultiplexedConnectionDriver {
    inner: vllm_quic_engine::MultiplexedConnectionDriver,
}

#[pymethods]
impl PyMultiplexedConnectionDriver {
    /// See `PyQuicConnectionDriver::connect_client`'s docstring - identical
    /// fd-handoff/blocking contract, just the multi-channel driver
    /// underneath.
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        fd, remote_host, remote_port, server_name, idle_timeout_ms,
        receive_window, send_window, stream_receive_window, max_congestion_window,
        max_message_bytes, handshake_timeout_ms,
    ))]
    fn connect_client(
        py: Python<'_>,
        fd: i32,
        remote_host: &str,
        remote_port: u16,
        server_name: &str,
        idle_timeout_ms: u32,
        receive_window: u64,
        send_window: u64,
        stream_receive_window: u64,
        max_congestion_window: u64,
        max_message_bytes: usize,
        handshake_timeout_ms: u64,
    ) -> PyResult<Self> {
        let socket = dup_socket(fd)?;
        let remote = parse_addr(remote_host, remote_port)?;
        let cfg = engine_config(
            idle_timeout_ms, receive_window, send_window, stream_receive_window,
            max_congestion_window, true,
        );
        let inner = py
            .detach(|| {
                vllm_quic_engine::MultiplexedConnectionDriver::connect_client(
                    socket, remote, server_name, cfg, max_message_bytes,
                    Duration::from_millis(handshake_timeout_ms),
                )
            })
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        fd, idle_timeout_ms, receive_window, send_window, stream_receive_window,
        max_congestion_window, max_message_bytes, handshake_timeout_ms,
    ))]
    fn connect_server(
        py: Python<'_>,
        fd: i32,
        idle_timeout_ms: u32,
        receive_window: u64,
        send_window: u64,
        stream_receive_window: u64,
        max_congestion_window: u64,
        max_message_bytes: usize,
        handshake_timeout_ms: u64,
    ) -> PyResult<Self> {
        let socket = dup_socket(fd)?;
        let cfg = engine_config(
            idle_timeout_ms, receive_window, send_window, stream_receive_window,
            max_congestion_window, false,
        );
        let inner = py
            .detach(|| {
                vllm_quic_engine::MultiplexedConnectionDriver::connect_server(
                    socket, cfg, max_message_bytes, Duration::from_millis(handshake_timeout_ms),
                )
            })
            .map_err(to_py_err)?;
        Ok(Self { inner })
    }

    /// Blocks (GIL released) until `data` has been fully handed to
    /// `channel`'s stream send buffer (opened lazily on first use), or
    /// `timeout_ms` elapses, or the connection is lost. Only one send may
    /// be in flight per channel at a time (matches every other backend's
    /// "send() blocks until handed off" contract - a caller that wants
    /// concurrent channels just needs one `send_on_channel` in flight per
    /// channel, not per connection).
    fn send_on_channel(&self, py: Python<'_>, channel: String, data: Vec<u8>, timeout_ms: u64) -> PyResult<()> {
        py.detach(|| self.inner.send_on_channel(channel, data, Duration::from_millis(timeout_ms)))
            .map_err(to_py_err)
    }

    /// Blocks (GIL released) for the next event on ANY channel - a
    /// complete message (returns `(channel, bytes)`) or the connection
    /// closing (raises). The caller is responsible for demultiplexing by
    /// the returned channel name into per-channel queues - see
    /// `quic_rs_broker.py`'s new design for where that happens on the
    /// Python side.
    fn recv_any<'py>(&self, py: Python<'py>, timeout_ms: u64) -> PyResult<(String, Bound<'py, pyo3::types::PyBytes>)> {
        let event = py
            .detach(|| self.inner.recv_any(Duration::from_millis(timeout_ms)))
            .map_err(to_py_err)?;
        match event {
            vllm_quic_engine::ChannelEvent::Message { channel, data } => {
                Ok((channel, pyo3::types::PyBytes::new(py, &data)))
            }
            vllm_quic_engine::ChannelEvent::Closed(reason) => {
                Err(to_py_err(format!("connection closed: {reason}")))
            }
        }
    }

    /// `&self`, not `&mut self` - see `PyQuicConnectionDriver::close`'s
    /// docstring for the real PyO3-borrow-conflict bug this avoids
    /// (must stay callable while a `recv_any()` call is in flight on
    /// another thread).
    fn close(&self, py: Python<'_>, drain_timeout_ms: u64) {
        py.detach(|| self.inner.close(Duration::from_millis(drain_timeout_ms)));
    }
}

/// Same `dup()`-then-`from_raw_fd` pattern as `udp_raw_engine`'s
/// `PyRawUdpEngine::from_fd` - see that crate's module docstring for the
/// full double-close/use-after-close hazard this avoids.
fn dup_socket(fd: i32) -> PyResult<std::net::UdpSocket> {
    use std::os::unix::io::FromRawFd;
    // SAFETY: `libc::dup` on a valid, open fd (the Python caller passes a
    // real `socket.socket().fileno()`) either returns a fresh valid fd or
    // -1 on error (checked immediately below).
    let dup_fd = unsafe { libc::dup(fd) };
    if dup_fd < 0 {
        return Err(to_py_err(std::io::Error::last_os_error()));
    }
    // SAFETY: `dup_fd` was just returned by a successful `dup()` above -
    // a fresh fd nothing else holds a Rust-side owner for yet.
    Ok(unsafe { std::net::UdpSocket::from_raw_fd(dup_fd) })
}

#[pyfunction]
fn ping() -> &'static str {
    vllm_quic_engine::ping()
}

#[pymodule]
fn _rust_quic_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_class::<PyQuicEngine>()?;
    m.add_class::<PyQuicConnectionDriver>()?;
    m.add_class::<PyMultiplexedConnectionDriver>()?;
    Ok(())
}
