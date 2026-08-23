// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Sans-io QUIC engine core (no PyO3, no networking code of its own).
//!
//! Built on `quinn-proto` per `/root/.claude/plans/frolicking-swimming-panda.md`
//! - see `engine.rs`'s module docstring for the real design (why both an
//! `Endpoint` and a `Connection` are needed even for exactly one peer, and
//! how bytes cross this crate's boundary with no socket ownership of its
//! own, mirroring `vllm/transport/quic_transport.py`'s relationship to
//! `aioquic`).

mod cert;
mod congestion;
mod driver;
mod engine;
mod error;
mod multiplexed_driver;

pub use driver::ConnectionDriver;
pub use engine::{Engine, EngineConfig, OutDatagram, StreamChunk, WindowConfig};
pub use error::{EngineError, Result};
pub use multiplexed_driver::{ChannelEvent, MultiplexedConnectionDriver};
pub use quinn_proto::{Dir, Event, StreamEvent, StreamId, VarInt};

/// Kept from Phase 0's build-pipeline proof - still exercised by the PyO3
/// crate's own smoke test, no reason to remove it.
pub fn ping() -> &'static str {
    "pong"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ping_returns_pong() {
        assert_eq!(ping(), "pong");
    }
}
