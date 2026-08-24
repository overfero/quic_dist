# flash-attn (Turing compatibility shim)

Presents [ssiu/flash-attention-turing](https://github.com/ssiu/flash-attention-turing)'s
real FlashAttention kernels for Turing GPUs (compute capability 7.5 -
T4, RTX 20-series) under the `flash_attn` package name and API surface
that `transformers`' `modeling_flash_attention_utils.py` imports
unconditionally once it detects a `flash_attn` package.

**Why this exists**: the official `flash-attn` PyPI package either
doesn't target Turing at all, or (as found directly in this project -
see `quic_dist/README.md`'s training infrastructure checklist) hits a
real C++ ABI mismatch against this environment's installed torch build
even when it compiles successfully. `ssiu/flash-attention-turing`'s
kernels were directly validated on a real T4 here: forward max diff
0.000488 / backward dQ max diff 0.001953 against `torch`'s own SDPA
(fp16 numerical noise, not a correctness bug), a real 1.27x forward
speedup, and a full end-to-end pass through both `transformers`'
regular `model.forward()` path AND this repo's own direct-decoder-layer
call pattern (`finetune.py`'s `run_decoder_layer` - real loss, real
backward, real gradients).

## Install

```bash
pip install einops "flash-attn-turing @ git+https://github.com/ssiu/flash-attention-turing.git"
cd flash_attn_turing_shim && pip install -e . --no-deps
```

`flash-attn-turing`'s own build takes a few minutes (compiles real CUDA
kernels). At runtime, torch's own `libc10.so` etc. need to be on the
dynamic linker's search path - a real, direct finding, not assumed:

```bash
export LD_LIBRARY_PATH=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")/lib:$LD_LIBRARY_PATH
```

## What's NOT supported

Carried over from the underlying kernels (see
`ssiu/flash-attention-turing`'s own README): no dropout, no
local/sliding-window attention, no KV cache. This shim raises a clear
`NotImplementedError` if a model/config actually needs one of these
(e.g. non-zero attention dropout, a sliding-window model) rather than
silently ignoring it - see `flash_attn/_hf_compat.py`.

## Usage

Set `attn_implementation: flash_attention_2` in any `PipelineConfig`
(finetune.py) - see `examples/configs/qwen25_0.5b_lora_flash_attn.yaml`
for a real, validated example.
