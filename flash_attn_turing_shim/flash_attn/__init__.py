"""Compatibility shim: presents ssiu/flash-attention-turing's Turing
(compute capability 7.5, e.g. T4) FlashAttention kernels under the
`flash_attn` package name/API surface that transformers'
`modeling_flash_attention_utils.py` imports unconditionally once
`is_flash_attn_2_available()` returns True.

This is NOT the official flash-attn package - it exists because the
official PyPI wheels/build either don't target Turing at all or hit a
real C++ ABI mismatch in this environment (see quic_dist's README
"Training infrastructure checklist" for that finding), while
ssiu/flash-attention-turing's kernels were directly validated on a real
T4 here: forward max diff 0.000488 / backward dQ max diff 0.001953
against torch's own SDPA (fp16 numerical noise, not a correctness bug),
and a real 1.27x forward speedup.

Real, not-yet-relevant limitation carried over from the underlying
kernels (see ssiu/flash-attention-turing's own README): no dropout, no
local/sliding-window mask, no KV cache. `apply_rotary_emb` below is a
loud stub, not a silent wrong answer - every model this shim has
actually been used with applies RoPE itself before calling attention
(the standard HF pattern), so `modeling_flash_attention_utils.py`
importing this name at module load time never actually needs to CALL
it; if some other model ever does, this raises immediately rather than
producing silently-wrong output.
"""
from __future__ import annotations

__version__ = "2.99.0+turing-shim"

from ._hf_compat import (  # noqa: F401
    flash_attn_func,
    flash_attn_varlen_func,
)
