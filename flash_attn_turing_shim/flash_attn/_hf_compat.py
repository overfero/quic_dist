"""Adapts transformers' calling convention (which always passes
dropout_p, and - once it sees this shim's __version__ >= 2.4.1 - also
deterministic/softcap) onto ssiu/flash-attention-turing's real, more
limited kernel signature (see that repo's own README: no dropout, no
local/sliding-window mask, no KV cache). Rejects loudly (not silently)
when a caller actually needs one of those unsupported features with a
non-default value - a silently-ignored dropout_p would be a real
correctness bug (the caller thinks dropout is being applied and it
isn't), not a missing convenience."""
from __future__ import annotations

from . import _turing_interface as _turing


def _check_unsupported(dropout_p: float, deterministic, softcap, window_size=None) -> None:
    if dropout_p not in (0.0, None):
        raise NotImplementedError(
            f"flash-attention-turing shim: dropout_p={dropout_p} requested, but the underlying "
            "ssiu/flash-attention-turing kernels don't support dropout (see its README's own "
            "'Does not support' list). Set attention dropout to 0 for this model/config, or use "
            "attn_implementation='sdpa' instead."
        )
    if softcap is not None:
        raise NotImplementedError(
            "flash-attention-turing shim: softcap requested (e.g. a Gemma2-style model), but the "
            "underlying kernels don't support it."
        )
    if window_size is not None and tuple(window_size) != (-1, -1):
        raise NotImplementedError(
            "flash-attention-turing shim: sliding-window attention requested, but the underlying "
            "kernels don't support local/window masks (see its README's 'Does not support' list)."
        )
    # deterministic is accepted and silently a no-op: the underlying kernels don't expose a
    # non-deterministic FAST PATH to opt out of in the first place (unlike official flash-attn,
    # which trades determinism for speed by default) - there is no correctness gap to warn about.


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                     deterministic=None, softcap=None, **_ignored):
    _check_unsupported(dropout_p, deterministic, softcap)
    return _turing.flash_attn_func(q, k, v, softmax_scale=softmax_scale, causal=causal)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                            dropout_p=0.0, softmax_scale=None, causal=False,
                            deterministic=None, softcap=None, **_ignored):
    _check_unsupported(dropout_p, deterministic, softcap)
    return _turing.flash_attn_varlen_func(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        softmax_scale=softmax_scale, causal=causal,
    )
