"""Loud stub - see flash_attn/__init__.py's module docstring for why
this is intentionally unimplemented rather than silently wrong: the
underlying ssiu/flash-attention-turing kernels don't provide a fused
rotary-embedding kernel, and every model this shim is actually used
with applies RoPE itself before calling attention, so
transformers/modeling_flash_attention_utils.py importing this NAME at
module load time never needs to actually CALL it in practice."""


def apply_rotary_emb(*args, **kwargs):
    raise NotImplementedError(
        "flash_attn.layers.rotary.apply_rotary_emb has no real implementation in the "
        "flash-attention-turing compatibility shim (see quic_dist/README.md's training "
        "infrastructure checklist) - if you're seeing this, something is calling flash_attn's "
        "own fused rotary embedding path, which this shim never intended to support."
    )
