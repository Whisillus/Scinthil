import math

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass.cute.runtime import from_dlpack

from .fa_api_sm120 import flash_attention_bshd_sm120


def flash_attention_bshd_torch_sm120(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run SM120 FlashAttention on BSHD Torch tensors."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must be rank-4 BSHD tensors")
    if not q.is_cuda or not k.is_cuda or not v.is_cuda:
        raise ValueError("q, k, and v must be CUDA tensors")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same CUDA device")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("q, k, and v must have the same FP16 or BF16 dtype")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("q, k, and v must have unit head-dimension stride")
    if q.data_ptr() % 16 != 0 or k.data_ptr() % 16 != 0 or v.data_ptr() % 16 != 0:
        raise ValueError("q, k, and v must be 16-byte aligned")

    batch, seqlen_q, head_q, headdim_qk = q.shape
    batch_k, seqlen_kv, head_kv, headdim_k = k.shape
    batch_v, seqlen_v, head_v, headdim_v = v.shape
    if batch_k != batch or batch_v != batch:
        raise ValueError("q, k, and v must have the same batch size")
    if seqlen_v != seqlen_kv or head_v != head_kv:
        raise ValueError("k and v must have the same sequence length and head count")
    if headdim_k != headdim_qk:
        raise ValueError("q and k must have the same head dimension")
    if head_q % head_kv != 0:
        raise ValueError("head_q must be divisible by head_kv")
    if headdim_qk % 64 != 0 or headdim_v % 64 != 0:
        raise ValueError("QK and V head dimensions must be divisible by 64")
    if min(batch, seqlen_q, seqlen_kv, head_q, head_kv) <= 0:
        raise ValueError("batch, sequence lengths, and head counts must be positive")

    scale = headdim_qk**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale):
        raise ValueError("softmax_scale must be finite")

    out = torch.empty((batch, seqlen_q, head_q, headdim_v), dtype=q.dtype, device=q.device)
    lse = torch.empty((batch, head_q, seqlen_q), dtype=torch.float32, device=q.device)

    with torch.cuda.device(q.device):
        stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)
        flash_attention_bshd_sm120(
            from_dlpack(q.detach(), assumed_align=16),
            from_dlpack(k.detach(), assumed_align=16),
            from_dlpack(v.detach(), assumed_align=16),
            from_dlpack(out, assumed_align=16),
            from_dlpack(lse, assumed_align=16),
            cutlass.Int32(seqlen_q),
            cutlass.Int32(seqlen_kv),
            cutlass.Float32(scale),
            bool(is_causal),
            stream,
        )
    return out, lse
