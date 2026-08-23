// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Phase 2 of `/root/.claude/plans/frolicking-swimming-panda.md`: two
//! `Engine`s (client + server role) over two real local UDP sockets, no
//! tokio/asyncio/PyO3 involved - proves a real TLS 1.3 handshake (including
//! `SkipServerVerification`, the self-signed-cert-skip-verification path)
//! completes deterministically before any Python integration is attempted.
//!
//! Waiting on real socket I/O (bounded by `set_read_timeout` plus an overall
//! deadline) is the synchronization point here, per `rust/AGENTS.md`'s
//! "prefer deterministic synchronization... sleep only as a last resort" -
//! there is no non-I/O observable signal available for a genuine network
//! round trip, so the read timeout stands in for it, not an arbitrary sleep.

use std::net::UdpSocket;
use std::time::{Duration, Instant};

use bytes::BytesMut;
use vllm_quic_engine::{Dir, Engine, EngineConfig, Event, OutDatagram, StreamEvent, WindowConfig};

const TEST_WINDOWS: WindowConfig = WindowConfig {
    receive_window: 1024 * 1024,
    send_window: 1024 * 1024,
    stream_receive_window: 1024 * 1024,
    max_congestion_window: 1024 * 1024,
};

fn engine_config(is_client: bool) -> EngineConfig {
    EngineConfig {
        is_client,
        idle_timeout_ms: 10_000,
        windows: TEST_WINDOWS,
    }
}

/// Pumps one socket's currently-available datagrams into the given engine,
/// returning whatever outgoing datagrams that produced - callers send those
/// on the SAME socket. Non-blocking-ish via a short read timeout: returns
/// as soon as the socket has nothing more to read right now, not on a fixed
/// delay.
fn pump_incoming(engine: &mut Engine, socket: &UdpSocket) -> Vec<OutDatagram> {
    let mut out = Vec::new();
    let mut buf = [0u8; 2048];
    loop {
        match socket.recv_from(&mut buf) {
            Ok((n, from)) => {
                let received = engine
                    .receive_datagram(Instant::now(), from, None, BytesMut::from(&buf[..n]))
                    .expect("receive_datagram must not error on a well-formed peer datagram");
                out.extend(received);
                out.extend(engine.poll_transmit(Instant::now()));
            }
            Err(error)
                if error.kind() == std::io::ErrorKind::WouldBlock
                    || error.kind() == std::io::ErrorKind::TimedOut =>
            {
                break;
            }
            Err(error) => panic!("unexpected socket error: {error}"),
        }
    }
    out
}

/// Test-only send path - just splits any GSO-batched `OutDatagram` back
/// into individual `send_to()` calls rather than actually using GSO, since
/// this test only cares about wire-format correctness, not syscall
/// efficiency (real GSO sending lives in `quic_rs_transport.py`'s
/// `_send_all`, exercised by the Python-level tests instead).
fn send_all(socket: &UdpSocket, datagrams: Vec<OutDatagram>) {
    for datagram in datagrams {
        match datagram.segment_size {
            None => {
                socket
                    .send_to(&datagram.data, datagram.addr)
                    .expect("send_to a loopback address must not fail in this test");
            }
            Some(segment_size) => {
                for chunk in datagram.data.chunks(segment_size) {
                    socket
                        .send_to(chunk, datagram.addr)
                        .expect("send_to a loopback address must not fail in this test");
                }
            }
        }
    }
}

fn drain_connected(engine: &mut Engine) -> bool {
    let mut connected = false;
    while let Some(event) = engine.poll_event() {
        if matches!(event, Event::Connected) {
            connected = true;
        }
    }
    connected
}

#[test]
fn handshake_completes_over_real_loopback_sockets() {
    let client_socket = UdpSocket::bind("127.0.0.1:0").unwrap();
    let server_socket = UdpSocket::bind("127.0.0.1:0").unwrap();
    client_socket
        .set_read_timeout(Some(Duration::from_millis(20)))
        .unwrap();
    server_socket
        .set_read_timeout(Some(Duration::from_millis(20)))
        .unwrap();
    let server_addr = server_socket.local_addr().unwrap();

    let (mut client_engine, initial) =
        Engine::new_client(&engine_config(true), server_addr, "vllm-pp-transport")
            .expect("client engine construction must succeed");
    send_all(&client_socket, initial);

    let mut server_engine =
        Engine::new_server(&engine_config(false)).expect("server engine construction must succeed");

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut client_connected = false;
    let mut server_connected = false;

    while Instant::now() < deadline && !(client_connected && server_connected) {
        let from_server = pump_incoming(&mut server_engine, &server_socket);
        send_all(&server_socket, from_server);
        let from_client = pump_incoming(&mut client_engine, &client_socket);
        send_all(&client_socket, from_client);

        client_connected |= drain_connected(&mut client_engine);
        server_connected |= drain_connected(&mut server_engine);
    }

    assert!(client_connected, "client side never observed Event::Connected");
    assert!(server_connected, "server side never observed Event::Connected");
}

#[test]
fn stream_messages_survive_out_of_order_datagram_delivery() {
    // Same handshake as above, then: open one persistent unidirectional
    // stream on the client, push several length-prefixed-by-the-caller
    // messages (framing is deliberately NOT this engine's job - see
    // Engine::write_stream's docstring), and confirm the server receives
    // every byte in order even when the underlying datagrams carrying them
    // are deliberately shuffled before delivery. This is the same property
    // test8_tensor_streaming.py needed for aioquic after finding the
    // original one-stream-per-message reordering bug - quinn-proto's own
    // per-stream ordering guarantee is what this test actually exercises,
    // not any code written in this crate.
    let client_socket = UdpSocket::bind("127.0.0.1:0").unwrap();
    let server_socket = UdpSocket::bind("127.0.0.1:0").unwrap();
    client_socket
        .set_read_timeout(Some(Duration::from_millis(20)))
        .unwrap();
    server_socket
        .set_read_timeout(Some(Duration::from_millis(20)))
        .unwrap();
    let server_addr = server_socket.local_addr().unwrap();

    let (mut client_engine, initial) =
        Engine::new_client(&engine_config(true), server_addr, "vllm-pp-transport").unwrap();
    send_all(&client_socket, initial);
    let mut server_engine = Engine::new_server(&engine_config(false)).unwrap();

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut client_connected = false;
    let mut server_connected = false;
    while Instant::now() < deadline && !(client_connected && server_connected) {
        let from_server = pump_incoming(&mut server_engine, &server_socket);
        send_all(&server_socket, from_server);
        let from_client = pump_incoming(&mut client_engine, &client_socket);
        send_all(&client_socket, from_client);
        client_connected |= drain_connected(&mut client_engine);
        server_connected |= drain_connected(&mut server_engine);
    }
    assert!(client_connected && server_connected, "handshake did not complete");

    let stream_id = client_engine.open_uni_stream().expect("open_uni_stream");
    let messages: Vec<Vec<u8>> = (0..20u32).map(|i| format!("msg-{i}").into_bytes()).collect();
    for message in &messages {
        client_engine
            .write_stream(stream_id, message)
            .expect("write_stream");
    }

    // Collect every outgoing datagram from the client WITHOUT sending it
    // yet, then deliver to the server in reverse order - proves reordering
    // at the UDP layer does not reorder the reassembled stream bytes.
    let mut pending: Vec<OutDatagram> = client_engine.poll_transmit(Instant::now());
    pending.reverse();
    for datagram in pending {
        // Same GSO-batch-aware split `send_all` uses - a batched
        // `OutDatagram` sent as one raw `send_to()` without splitting on
        // `segment_size` would arrive as one oversized malformed datagram
        // instead of several real ones.
        let segments: Vec<&[u8]> = match datagram.segment_size {
            None => vec![&datagram.data[..]],
            Some(segment_size) => datagram.data.chunks(segment_size).collect(),
        };
        for segment in segments {
            server_socket.send_to(segment, datagram.addr).unwrap();
            // A tiny gap between sends so the shuffled arrival order is not
            // an artifact of the OS batching them back into original order
            // on a loopback interface.
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut received: Vec<u8> = Vec::new();
    let mut server_stream_id = None;
    while Instant::now() < deadline {
        let _ = pump_incoming(&mut server_engine, &server_socket);
        while let Some(event) = server_engine.poll_event() {
            if let Event::Stream(StreamEvent::Opened { dir: Dir::Uni }) = event {
                while let Some(id) = server_engine.accept_uni_stream().unwrap() {
                    server_stream_id = Some(id);
                }
            }
        }
        if let Some(id) = server_stream_id {
            for chunk in server_engine.read_stream(id).expect("read_stream") {
                received.extend_from_slice(&chunk.bytes);
            }
        }
        if received.len() >= messages.iter().map(|m| m.len()).sum::<usize>() {
            break;
        }
    }

    let expected: Vec<u8> = messages.concat();
    assert_eq!(
        received, expected,
        "reassembled stream bytes must match send() order exactly, regardless of UDP delivery order"
    );
}
