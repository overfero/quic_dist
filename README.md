# quic_dist

A real `torch.distributed.ProcessGroup`/`Store` backend over QUIC, registered
as `"quic"`. Built for pipeline-parallel training (LoRA, QLoRA, or full
fine-tuning) between machines that are behind NAT, with no cloud VPC, no
port forwarding, and no direct reachability — only that every rank can reach
one shared HTTP signaling server.

Standalone: this package has no dependency on vLLM or any other project.
It vendors its own copy of the UDP hole-punch client (`holepunch/peer.py`)
and its own small Rust workspace (`rust/`) for the QUIC engine.

## Install

```bash
git clone <this-repo-url> quic_dist   # directory name doesn't matter once installed
cd quic_dist
pip install -e .                      # installs the `quic_dist` package + Python deps

cd rust && cargo build --release      # builds the compiled QUIC engine
cp target/release/lib_rust_quic_engine.so ../_rust_quic_engine.abi3.so
cd ..
```

Requires a Rust toolchain (matching `rust/rust-toolchain.toml`, installed
automatically via [rustup](https://rustup.rs) if you don't already have one)
to build the QUIC engine. If you only need the signaling server, install
with `pip install -e ".[signaling]"` for its `fastapi`/`uvicorn` deps.

## Usage

```python
import quic_dist

quic_dist.init_process_group(
    signaling_url="https://your-signaling-server",
    rank=rank,
    world_size=world_size,
)

# from here on, standard torch.distributed - nothing quic_dist-specific:
import torch.distributed as dist
dist.send(tensor, dst=1, tag=7)
dist.recv(tensor, src=1, tag=7)
dist.barrier()
```

Run the signaling server (needed once, reachable by every rank):

```bash
cd holepunch
uvicorn signaling_server:app --host 0.0.0.0 --port 8000
```

## Scope

Point-to-point communication only: `send`/`recv`/`isend`/`irecv`/`barrier`.
Every collective (`all_reduce`, `all_gather`, `broadcast`, ...) raises
`NotImplementedError` by explicit design — not a silent no-op, not a
fallback to another backend.

## Tests

```bash
cd tests
pip install pytest
python3 -m pytest test_store.py test_process_group.py test_pipeline.py test_parallel_stream.py -q
```

These spin up a real local signaling server and real hole-punched QUIC
connections (loopback) — no mocking. `cross_machine_rank.py`,
`lora_pipeline_rank.py`, `real_llm_pipeline_rank.py`, and
`vision_shape_rank.py` are standalone scripts for real cross-machine
testing (run one instance per machine, pointed at a real publicly
reachable signaling URL).

## Known limitations

See the full deliverable report for the complete architecture, design
rationale, and an honest list of what's implemented vs. not:
https://claude.ai/code/artifact/96117257-82d9-476a-ade9-aba41a56b8f4
