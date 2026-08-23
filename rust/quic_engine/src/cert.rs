// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! TLS certificate handling: a fresh self-signed cert per process for the
//! server role, and a "skip verification" client-side `ServerCertVerifier`
//! for the client role - the exact same trust model
//! `vllm/transport/quic_transport.py` already uses for aioquic (no real
//! peer authentication in v1: integrity/confidentiality-in-transit only,
//! from TLS 1.3 itself, not "is this actually who I think it is" - anyone
//! who completed the same UDP hole punch can also complete this handshake).
//! See that module's docstring, "Known limitations", for the full rationale
//! - unchanged here, just ported to Rust.

use std::sync::Arc;

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::CryptoProvider;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, SignatureScheme};

use crate::error::{config, Result};

/// A fresh, self-signed certificate + private key, generated per process -
/// mirrors `quic_transport.py`'s `_generate_self_signed_cert()`.
pub fn generate_self_signed_cert() -> Result<(CertificateDer<'static>, PrivateKeyDer<'static>)> {
    let generated = rcgen::generate_simple_self_signed(vec!["vllm-pp-transport".to_string()])
        .map_err(|error| config!("failed to generate self-signed certificate: {error}"))?;
    let cert_der = generated.cert.der().clone();
    let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(
        generated.key_pair.serialize_der(),
    ));
    Ok((cert_der, key_der))
}

/// Skips server certificate verification entirely - see module docstring.
/// Ported from quinn's own `insecure_connection.rs` example (the canonical
/// pattern for this with `rustls::client::danger::ServerCertVerifier`).
#[derive(Debug)]
pub struct SkipServerVerification(Arc<CryptoProvider>);

impl SkipServerVerification {
    pub fn new(provider: Arc<CryptoProvider>) -> Arc<Self> {
        Arc::new(Self(provider))
    }
}

impl ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> std::result::Result<ServerCertVerified, rustls::Error> {
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls12_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
        rustls::crypto::verify_tls13_signature(
            message,
            cert,
            dss,
            &self.0.signature_verification_algorithms,
        )
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.0.signature_verification_algorithms.supported_schemes()
    }
}
