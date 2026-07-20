from typing import Any

import cuda.bindings.runtime as cuda_runtime
import cutlass
import cutlass.cute as cute

from ..utils import CudaTensor, TensorLayout, check_cuda_runtime


def get_fa_metrics(
    *,
    q_layout: TensorLayout,
    k_layout: TensorLayout,
    v_layout: TensorLayout,
    dtype: type[Any],
    is_causal: bool,
) -> tuple[int, int, int]:
    """Return Q/K/V load bytes, O/LSE store bytes, and QK+PV matmul-equivalent FLOPs."""
    if dtype not in (cutlass.Float16, cutlass.BFloat16):
        raise TypeError("FlashAttention inputs must use FP16 or BF16")
    if q_layout.rank != 4 or k_layout.rank != 4 or v_layout.rank != 4:
        raise ValueError("q_layout, k_layout, and v_layout must describe rank-4 BSHD tensors")

    batch, seqlen_q, head_q, headdim_qk = q_layout.shape
    batch_k, seqlen_kv, head_kv, headdim_k = k_layout.shape
    batch_v, seqlen_v, head_v, headdim_v = v_layout.shape
    if batch_k != batch or batch_v != batch:
        raise ValueError("Q, K, and V layouts must have the same batch extent")
    if seqlen_v != seqlen_kv or head_v != head_kv:
        raise ValueError("K and V layouts must have matching sequence and head extents")
    if headdim_k != headdim_qk:
        raise ValueError("Q and K layouts must have the same head dimension")
    if head_q % head_kv != 0:
        raise ValueError("Q head count must be divisible by K/V head count")

    active_pairs = seqlen_q * seqlen_kv
    if is_causal:
        kv_q_offset = seqlen_kv - seqlen_q
        active_pairs = sum(max(0, min(seqlen_kv, row + kv_q_offset + 1)) for row in range(seqlen_q))

    load_bytes = (q_layout.cosize + k_layout.cosize + v_layout.cosize) * dtype.width // 8
    store_bytes = batch * seqlen_q * head_q * headdim_v * dtype.width // 8
    store_bytes += batch * head_q * seqlen_q * cutlass.Float32.width // 8
    flops = 2 * batch * head_q * active_pairs * (headdim_qk + headdim_v)
    return load_bytes, store_bytes, flops


def get_input_Q(
    *,
    q_layout: TensorLayout,
    dtype: type[Any],
) -> tuple[CudaTensor, int, None, None]:
    """Allocate BSHD Q and return fixed-length sequence metadata."""
    if dtype not in (cutlass.Float16, cutlass.BFloat16):
        raise TypeError("FlashAttention inputs must use FP16 or BF16")
    if q_layout.rank != 4:
        raise ValueError("q_layout must describe a rank-4 BSHD tensor")

    num_bytes = q_layout.cosize * dtype.width // 8
    address = check_cuda_runtime(cuda_runtime.cudaMalloc(num_bytes))
    ptr = cute.runtime.make_ptr(
        dtype,
        address,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )

    mQ = CudaTensor(ptr, q_layout, address)
    max_seqlen_q = q_layout.shape[1]
    return mQ, max_seqlen_q, None, None


def get_input_KV(
    *,
    k_layout: TensorLayout,
    v_layout: TensorLayout,
    dtype: type[Any],
) -> tuple[CudaTensor, CudaTensor, int, None, None]:
    """Allocate BSHD K/V and return their shared sequence metadata."""
    if dtype not in (cutlass.Float16, cutlass.BFloat16):
        raise TypeError("FlashAttention inputs must use FP16 or BF16")
    if k_layout.rank != 4 or v_layout.rank != 4:
        raise ValueError("k_layout and v_layout must describe rank-4 BSHD tensors")
    for mode in (0, 1, 2):
        if k_layout.shape[mode] != v_layout.shape[mode]:
            raise ValueError("K and V layouts must have matching batch, sequence, and head extents")

    k_num_bytes = k_layout.cosize * dtype.width // 8
    v_num_bytes = v_layout.cosize * dtype.width // 8

    k_address = check_cuda_runtime(cuda_runtime.cudaMalloc(k_num_bytes))
    v_address = check_cuda_runtime(cuda_runtime.cudaMalloc(v_num_bytes))
    k_ptr = cute.runtime.make_ptr(
        dtype,
        k_address,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    v_ptr = cute.runtime.make_ptr(
        dtype,
        v_address,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )

    mK = CudaTensor(k_ptr, k_layout, k_address)
    mV = CudaTensor(v_ptr, v_layout, v_address)
    max_seqlen_kv = k_layout.shape[1]
    return mK, mV, max_seqlen_kv, None, None


def get_output_O(
    *,
    o_layout: TensorLayout,
    dtype: type[Any],
) -> CudaTensor:
    """Allocate O using a caller-provided layout."""
    if dtype not in (cutlass.Float16, cutlass.BFloat16):
        raise TypeError("FlashAttention output must use FP16 or BF16")
    if o_layout.rank != 4:
        raise ValueError("o_layout must describe a rank-4 BSHD tensor")

    o_num_bytes = o_layout.cosize * dtype.width // 8

    o_address = check_cuda_runtime(cuda_runtime.cudaMalloc(o_num_bytes))
    o_ptr = cute.runtime.make_ptr(
        dtype,
        o_address,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return CudaTensor(o_ptr, o_layout, o_address)


def get_output_LSE(*, lse_layout: TensorLayout) -> CudaTensor:
    """Allocate FP32 LSE using a caller-provided layout."""
    if lse_layout.rank != 3:
        raise ValueError("lse_layout must describe a rank-3 BHS tensor")

    lse_num_bytes = lse_layout.cosize * cutlass.Float32.width // 8

    lse_address = check_cuda_runtime(cuda_runtime.cudaMalloc(lse_num_bytes))
    lse_ptr = cute.runtime.make_ptr(
        cutlass.Float32,
        lse_address,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    return CudaTensor(lse_ptr, lse_layout, lse_address)


__all__ = [
    "get_fa_metrics",
    "get_input_KV",
    "get_input_Q",
    "get_output_LSE",
    "get_output_O",
]
