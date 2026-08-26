#!/bin/bash
# One-time setup for a quic-train (training) machine. Idempotent - safe
# to re-run. Real recipe learned recovering from a full environment
# reset (see this repo's README's own Install section for the
# unmodified base version of this) - captures every step that turned
# out to actually be needed, not just the happy path.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

echo "=== [1/4] quic_dist package + Python deps ==="
pip install -e . -q
pip install -e ".[finetune]" -q   # transformers, peft, bitsandbytes, accelerate, datasets, pyyaml

echo "=== [2/4] Rust QUIC engine ==="
if ! command -v cargo &>/dev/null; then
    echo "rustup not found, installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
    source "$HOME/.cargo/env"
fi
source "$HOME/.cargo/env" 2>/dev/null || true
(cd rust && cargo build --release)
cp rust/target/release/lib_rust_quic_engine.so _rust_quic_engine.abi3.so

echo "=== [3/4] verify the Rust engine actually imports ==="
python3 -c "import _rust_quic_engine; print('quic_dist Rust engine OK')"

echo "=== [4/4] done ==="
echo "quic_dist is ready. Next: download a base model into /data/models/ and"
echo "point GRPOConfig.model_path at it - see README's own GRPO section."
