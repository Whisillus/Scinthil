from typing import Any

import cutlass
import torch


def m_grouped_masked_gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("a and b must be rank-3 tensors")
    if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[2]:
        raise ValueError("a and b must have compatible grouped GEMM shapes")
    if out_dtype is None:
        out_dtype = a.dtype
    return torch.einsum("gmk,gnk->gmn", a.float(), b.float()).to(out_dtype)


def get_m_grouped_gemm_masked_metrics(
    *,
    groups: int,
    max_m: int,
    m: int,
    n: int,
    k: int,
    dtype: type[Any],
) -> tuple[int, int, int]:
    """Return logical A/B load bytes, D store bytes, and matmul FLOPs."""
    if dtype not in (cutlass.Float16, cutlass.BFloat16):
        raise TypeError("grouped GEMM inputs must use FP16 or BF16")
    if min(groups, max_m, m, n, k) <= 0:
        raise ValueError("groups, max_m, M, N, and K must be positive")
    if m > max_m:
        raise ValueError("M must not exceed max_m")

    element_bytes = dtype.width // 8
    valid_m = groups * m
    load_bytes = (valid_m * k + groups * n * k) * element_bytes
    store_bytes = valid_m * n * element_bytes
    flops = 2 * valid_m * n * k
    return load_bytes, store_bytes, flops
