from typing import Any

import cutlass

from ..utils import TensorLayout


def get_flash_attention_metrics(
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
