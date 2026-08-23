// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! A congestion-window ceiling, layered on top of any other
//! `quinn_proto::congestion::Controller`.
//!
//! Found by real testing (see `quic_rs_transport.py`'s module docstring,
//! "KNOWN, UNRESOLVED LIMITATION"): a single message larger than roughly
//! 5-10MB reliably stalled the connection completely. Root cause, confirmed
//! by reading `quinn-proto`'s own source rather than guessing: this
//! project's deployment machines cap `net.core.rmem_max`/`wmem_max` at
//! 212992 bytes (208 KiB, confirmed via `sysctl` on both machines, see the
//! `project_os_udp_buffer_ceiling` memory entry) and cannot raise it from
//! inside the container. The configured controller (`CubicConfig` as of
//! `engine.rs::build_transport_config` - this file originally reasoned
//! about `NewReno`, since corrected; both share the same property below)
//! is free to grow its congestion window arbitrarily large for a
//! sustained transfer - neither `quinn_proto::congestion::NewRenoConfig`
//! nor `CubicConfig` exposes a maximum-window cap (confirmed by reading
//! `congestion/new_reno.rs` and `congestion/cubic.rs` directly: `window`
//! only ever grows on ACK, or resets down on loss - there is no ceiling
//! field in either). Once
//! the window grows past what the OS socket buffers can actually hold for a
//! large enough transfer, the connection bursts more data onto the wire
//! than the kernel can buffer, causing self-induced packet loss at the
//! socket layer - invisible to and unrelated to any QUIC-level flow-control
//! window (`send_window`/`receive_window`/`stream_receive_window`), which is
//! why raising those (tested up to 200 MiB) did not help even slightly.
//!
//! The fix here does not depend on network RTT happening to pace bursts
//! naturally (which only helps on a real, non-loopback link, and was never
//! confirmed - see `quic_rs_transport.py`'s module docstring for the
//! abandoned cross-machine test attempt): it clamps the reported congestion
//! window
//! itself, at the source, regardless of how large the wrapped controller's
//! internal state grows. `quinn-proto`'s own `write_limit()` computation
//! (`connection/streams/state.rs`) already gates every write against
//! `Controller::window()` amongst other things, so capping what this method
//! *returns* is sufficient to cap how much can ever be in flight
//! unacknowledged - no change to `write_limit()` itself, or to the wrapped
//! controller's internal slow-start/congestion-avoidance/loss-recovery
//! logic, is needed.
use std::any::Any;
use std::sync::Arc;
use std::time::Instant;

use quinn_proto::RttEstimator;
use quinn_proto::congestion::{Controller, ControllerFactory, ControllerMetrics};

/// Wraps any `Controller`, capping `window()`/`initial_window()` at
/// `max_window` without altering the wrapped controller's own bookkeeping
/// (on_sent/on_ack/on_congestion_event etc. all pass straight through) - the
/// wrapped controller can still believe its window grew past `max_window`
/// internally, it just never gets reported past the cap, so it also never
/// authorizes more than `max_window` bytes of in-flight data.
pub struct BoundedController {
    inner: Box<dyn Controller>,
    max_window: u64,
}

impl Controller for BoundedController {
    fn on_sent(&mut self, now: Instant, bytes: u64, last_packet_number: u64) {
        self.inner.on_sent(now, bytes, last_packet_number);
    }

    fn on_ack(
        &mut self,
        now: Instant,
        sent: Instant,
        bytes: u64,
        app_limited: bool,
        rtt: &RttEstimator,
    ) {
        self.inner.on_ack(now, sent, bytes, app_limited, rtt);
    }

    fn on_end_acks(
        &mut self,
        now: Instant,
        in_flight: u64,
        app_limited: bool,
        largest_packet_num_acked: Option<u64>,
    ) {
        self.inner
            .on_end_acks(now, in_flight, app_limited, largest_packet_num_acked);
    }

    fn on_congestion_event(
        &mut self,
        now: Instant,
        sent: Instant,
        is_persistent_congestion: bool,
        lost_bytes: u64,
    ) {
        self.inner
            .on_congestion_event(now, sent, is_persistent_congestion, lost_bytes);
    }

    fn on_mtu_update(&mut self, new_mtu: u16) {
        self.inner.on_mtu_update(new_mtu);
    }

    fn window(&self) -> u64 {
        self.inner.window().min(self.max_window)
    }

    fn metrics(&self) -> ControllerMetrics {
        let mut metrics = self.inner.metrics();
        metrics.congestion_window = metrics.congestion_window.min(self.max_window);
        metrics
    }

    fn clone_box(&self) -> Box<dyn Controller> {
        Box::new(BoundedController {
            inner: self.inner.clone_box(),
            max_window: self.max_window,
        })
    }

    fn initial_window(&self) -> u64 {
        self.inner.initial_window().min(self.max_window)
    }

    fn into_any(self: Box<Self>) -> Box<dyn Any> {
        self
    }
}

/// Builds a `BoundedController` around whatever `inner` factory produces
/// (`NewRenoConfig`, `CubicConfig`, `BbrConfig`, ...) - generic so the
/// underlying algorithm can still be swapped later without touching this
/// file.
pub struct BoundedControllerFactory<F> {
    inner: Arc<F>,
    max_window: u64,
}

impl<F> BoundedControllerFactory<F> {
    pub fn new(inner: Arc<F>, max_window: u64) -> Self {
        Self { inner, max_window }
    }
}

impl<F: ControllerFactory + Send + Sync + 'static> ControllerFactory for BoundedControllerFactory<F> {
    fn build(self: Arc<Self>, now: Instant, current_mtu: u16) -> Box<dyn Controller> {
        Box::new(BoundedController {
            inner: self.inner.clone().build(now, current_mtu),
            max_window: self.max_window,
        })
    }
}
