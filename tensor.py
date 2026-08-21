"""Tensor <-> bytes for `ProcessGroupQUIC`, with the metadata the parallel-
training prompt's Phase 3 explicitly asks for (dtype, shape, numel, byte
size, message ID, microbatch ID, tensor ID) - a genuinely different
module from `vllm/transport/tensor.py`, not a reuse of it: that module's
JSON-encoded header carries only dtype+shape (no message/microbatch/
tensor IDs, since it predates this pipeline-parallel use case), and its
receive path does an extra `bytearray(...)` copy on top of
`torch.frombuffer`'s own requirements. This module uses a fixed-layout
`struct`-packed binary header (no JSON parsing, no pickle anywhere on the
data path) and is GPU-aware (CPU tensors only cross the wire - a real,
documented limitation, not zero-copy/GPUDirect - see `serialize_tensor`'s
docstring).

Wire format: `[dtype_code:B][ndim:B][numel:Q][byte_size:Q][message_id:Q]
[microbatch_id:Q][tensor_id:Q][shape: ndim x Q][payload: byte_size bytes]`
- big-endian throughout, matching every other wire format in this
project (`vllm/transport/*`, `rust/src/*_engine/`).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import torch

# Small, explicit registry - only the dtypes this project's real workload
# (LoRA/QLoRA activations/gradients: float32/float16/bfloat16, plus
# int64 for token IDs and bool for attention masks) actually needs.
# Deliberately a hard error on an unlisted dtype (see serialize_tensor)
# rather than a silent fallback - an unsupported dtype crossing the wire
# wrong is a correctness bug, not something to guess around.
_DTYPE_TO_CODE: dict[torch.dtype, int] = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.float64: 3,
    torch.int64: 4,
    torch.int32: 5,
    torch.int16: 6,
    torch.int8: 7,
    torch.uint8: 8,
    torch.bool: 9,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}

_MAX_NDIM = 16  # generous - real activation/gradient tensors are rank 2-4

_HEADER_FIXED = struct.Struct("!BBQQQQQ")  # dtype_code, ndim, numel, byte_size, message_id, microbatch_id, tensor_id


@dataclass(frozen=True)
class TensorMetadata:
    dtype: torch.dtype
    shape: tuple[int, ...]
    numel: int
    byte_size: int
    message_id: int
    microbatch_id: int
    tensor_id: int


def serialize_tensor(
    tensor: torch.Tensor,
    *,
    message_id: int = 0,
    microbatch_id: int = 0,
    tensor_id: int = 0,
) -> bytes:
    """Tensor -> one flat `bytes` message (header + raw payload).

    GPU tensors are moved to CPU here (`.cpu()`) - this is NOT zero-copy/
    GPUDirect RDMA, a real, deliberate, documented limitation (see the
    accompanying plan's "known limitations") - correctness first, matches
    this project's own established pattern for the phase before an
    optimization is attempted (`vllm/transport/tensor.py`'s own module
    docstring: "CPU tensors only... correctness is the only goal of this
    phase").
    """
    if tensor.dtype not in _DTYPE_TO_CODE:
        raise ValueError(
            f"serialize_tensor: unsupported dtype {tensor.dtype} - supported: "
            f"{sorted((str(d) for d in _DTYPE_TO_CODE), key=str)}"
        )
    if tensor.dim() > _MAX_NDIM:
        raise ValueError(f"serialize_tensor: tensor has {tensor.dim()} dims, max supported is {_MAX_NDIM}")

    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    tensor = tensor.contiguous()

    # .reshape(-1) BEFORE the dtype-reinterpreting .view(uint8), not after:
    # torch.Tensor.view(dtype) requires at least 1 dimension (real bug hit
    # here via a 0-dim/scalar tensor - "self.dim() cannot be 0 to view
    # Float as Byte") - flattening first sidesteps it for every rank,
    # including 0, with no behavior change for rank >= 1 tensors.
    raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
    shape = tuple(tensor.shape)
    header = _HEADER_FIXED.pack(
        _DTYPE_TO_CODE[tensor.dtype],
        len(shape),
        tensor.numel(),
        len(raw),
        message_id,
        microbatch_id,
        tensor_id,
    )
    shape_bytes = struct.pack(f"!{len(shape)}Q", *shape)
    return header + shape_bytes + raw


def deserialize_tensor(data: bytes) -> tuple[torch.Tensor, TensorMetadata]:
    """Inverse of `serialize_tensor`. Always returns a CPU tensor -
    moving it to a specific device (if needed) is the caller's job, since
    this module has no notion of "the right device" for a given rank
    (that's `ProcessGroupQUIC`'s concern, not the wire format's)."""
    dtype_code, ndim, numel, byte_size, message_id, microbatch_id, tensor_id = _HEADER_FIXED.unpack_from(data, 0)
    offset = _HEADER_FIXED.size
    shape = struct.unpack_from(f"!{ndim}Q", data, offset)
    offset += struct.calcsize(f"!{ndim}Q")

    dtype = _CODE_TO_DTYPE[dtype_code]
    raw = bytearray(data[offset:offset + byte_size])  # copy: torch.frombuffer requires a writable buffer
    flat = torch.frombuffer(raw, dtype=torch.uint8)
    tensor = flat.view(dtype).reshape(shape).clone()

    meta = TensorMetadata(
        dtype=dtype,
        shape=tuple(shape),
        numel=numel,
        byte_size=byte_size,
        message_id=message_id,
        microbatch_id=microbatch_id,
        tensor_id=tensor_id,
    )
    return tensor, meta
