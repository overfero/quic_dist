// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Error type for `vllm_quic_engine`.
//!
//! Every fallible operation in this crate returns `Result<_, EngineError>` -
//! see `rust/AGENTS.md` for why (`.unwrap()`/`panic!` on network-controlled
//! input must never happen here: the workspace builds with `panic = "abort"`
//! in both dev and release profiles, so a panic crossing the PyO3 boundary
//! would abort the whole Python process, not raise a catchable exception -
//! strictly worse than aioquic's behavior on malformed packets, which this
//! engine is meant to replace, not regress).

use thiserror::Error;
use thiserror_ext::Macro;

/// Result alias for engine operations.
pub type Result<T> = std::result::Result<T, EngineError>;

/// Errors produced while configuring or driving a QUIC `Engine`.
#[derive(Debug, Error, Macro)]
#[thiserror_ext(macro(path = "crate::error"))]
pub enum EngineError {
    #[error("QUIC engine configuration failed: {message}")]
    Config { message: String },
    #[error("QUIC connect failed: {message}")]
    Connect { message: String },
    #[error("QUIC protocol error: {message}")]
    Protocol { message: String },
    #[error("QUIC stream error: {message}")]
    Stream { message: String },
    #[error("no active connection - the engine has not finished a handshake yet")]
    NoConnection,
    #[error("QUIC socket I/O error: {message}")]
    Io { message: String },
    #[error("QUIC connection closed: {message}")]
    Closed { message: String },
    #[error("QUIC operation timed out: {message}")]
    Timeout { message: String },
}
