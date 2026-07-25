from typing import Any

import cutlass
import cutlass.cute as cute
import torch


def get_m_grouped_masked_gemm_deepgemm_case() -> tuple[tuple[int, int, int, int, int], ...]:
    """Return DeepGEMM's masked grouped GEMM cases as (G, Mmax, M, N, K)."""
    return (
        (32, 4096, 192, 6144, 7168),
        (32, 4096, 192, 7168, 3072),
        (32, 4096, 192, 4096, 4096),
        (32, 4096, 192, 4096, 2048),
        (6, 4096, 1024, 6144, 7168),
        (6, 4096, 1024, 7168, 3072),
        (6, 4096, 1024, 4096, 4096),
        (6, 4096, 1024, 4096, 2048),
        (32, 4096, 20, 6144, 7168),
        (32, 4096, 20, 7168, 3072),
        (32, 4096, 20, 4096, 4096),
        (32, 4096, 20, 4096, 2048),
        (6, 4096, 20, 6144, 7168),
        (6, 4096, 20, 7168, 3072),
        (6, 4096, 20, 4096, 4096),
        (6, 4096, 20, 4096, 2048),
    )


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


def get_m_grouped_gemm_masked_fp8_metrics(
    *,
    groups: int,
    max_m: int,
    m: int,
    n: int,
    k: int,
) -> tuple[int, int, int]:
    """Return logical FP8 operand/scale load bytes, BF16 store bytes, and matmul FLOPs."""
    if min(groups, max_m, m, n, k) <= 0:
        raise ValueError("groups, max_m, M, N, and K must be positive")
    if m > max_m:
        raise ValueError("M must not exceed max_m")
    if k % 128 != 0:
        raise ValueError("K must be divisible by 128")

    valid_m = groups * m
    k_scale_blocks = k // 128
    n_scale_blocks = cute.ceil_div(n, 128)
    operand_load_bytes = (valid_m * k + groups * n * k) * (cutlass.Float8E4M3FN.width // 8)
    scale_load_bytes = (valid_m * k_scale_blocks + groups * n_scale_blocks * k_scale_blocks) * (
        cutlass.Float32.width // 8
    )
    store_bytes = valid_m * n * (cutlass.BFloat16.width // 8)
    flops = 2 * valid_m * n * k
    return operand_load_bytes + scale_load_bytes, store_bytes, flops
