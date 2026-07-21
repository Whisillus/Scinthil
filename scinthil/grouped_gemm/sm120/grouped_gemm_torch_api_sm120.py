import cuda.bindings.driver as cuda
import torch
from cutlass.cute.runtime import from_dlpack

from .grouped_gemm_api_sm120 import (
    get_m_grouped_masked_gemm_fp8_tile,
    get_m_grouped_masked_gemm_tile,
    m_grouped_masked_gemm_fp8_sm120,
    m_grouped_masked_gemm_sm120,
)


def m_grouped_masked_gemm_torch_sm120(
    a: torch.Tensor,
    b: torch.Tensor,
    token_info: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
    use_block_swizzle: bool = True,
    block_swizzle_factor: int = 8,
) -> torch.Tensor:
    """Run SM120 M-grouped masked GEMM on dense grouped Torch tensors."""
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("a and b must be rank-3 tensors")
    if token_info.ndim != 1:
        raise ValueError("token_info must be a rank-1 tensor")
    if not a.is_cuda or not b.is_cuda or not token_info.is_cuda:
        raise ValueError("a, b, and token_info must be CUDA tensors")
    if a.device != b.device or a.device != token_info.device:
        raise ValueError("a, b, and token_info must be on the same CUDA device")
    if a.dtype not in (torch.float16, torch.bfloat16) or b.dtype != a.dtype:
        raise TypeError("a and b must have the same FP16 or BF16 dtype")
    if token_info.dtype != torch.int32:
        raise TypeError("token_info must have dtype torch.int32")
    if not a.is_contiguous() or not b.is_contiguous() or not token_info.is_contiguous():
        raise ValueError("a, b, and token_info must be contiguous")
    if a.data_ptr() % 16 != 0 or b.data_ptr() % 16 != 0 or token_info.data_ptr() % 16 != 0:
        raise ValueError("a, b, and token_info must be 16-byte aligned")

    groups, max_m, gemm_k = a.shape
    groups_b, n, gemm_k_b = b.shape
    if groups_b != groups:
        raise ValueError("a and b must have the same group count")
    if gemm_k_b != gemm_k:
        raise ValueError("a and b must have the same K dimension")
    if token_info.shape != (groups,):
        raise ValueError("token_info must have shape [G]")
    if min(groups, max_m, n, gemm_k) <= 0:
        raise ValueError("G, Mmax, N, and K must be positive")
    _, _, tile_k = get_m_grouped_masked_gemm_tile()
    if gemm_k % tile_k != 0:
        raise ValueError(f"K must be divisible by {tile_k}")
    if block_swizzle_factor <= 0:
        raise ValueError("block_swizzle_factor must be positive")

    capability = torch.cuda.get_device_capability(a.device)
    if capability != (12, 0):
        raise RuntimeError(f"SM120 CUDA device is required, got capability {capability}")

    if out is None:
        out = torch.empty((groups, max_m, n), dtype=a.dtype, device=a.device)
    else:
        if out.ndim != 3:
            raise ValueError("out must be a rank-3 tensor")
        if not out.is_cuda:
            raise ValueError("out must be a CUDA tensor")
        if out.device != a.device:
            raise ValueError("out must be on the same CUDA device as a")
        if out.dtype != a.dtype:
            raise TypeError("out must have the same dtype as a")
        if out.shape != (groups, max_m, n):
            raise ValueError("out must have shape [G, Mmax, N]")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if out.data_ptr() % 16 != 0:
            raise ValueError("out must be 16-byte aligned")

    with torch.cuda.device(a.device):
        stream = cuda.CUstream(torch.cuda.current_stream(device=a.device).cuda_stream)
        m_grouped_masked_gemm_sm120(
            from_dlpack(a.detach(), assumed_align=16),
            from_dlpack(b.detach(), assumed_align=16),
            from_dlpack(out, assumed_align=16),
            from_dlpack(token_info.detach(), assumed_align=16),
            stream,
            bool(use_block_swizzle),
            int(block_swizzle_factor),
        )
    return out


def m_grouped_masked_gemm_fp8_torch_sm120(
    a: torch.Tensor,
    scale_a: torch.Tensor,
    b: torch.Tensor,
    scale_b: torch.Tensor,
    token_info: torch.Tensor,
    recipe_a: tuple[int, int] = (1, 128),
    recipe_b: tuple[int, int] = (128, 128),
    *,
    out: torch.Tensor | None = None,
    use_block_swizzle: bool = True,
    block_swizzle_factor: int = 8,
) -> torch.Tensor:
    """Run groupwise-scaled SM120 FP8 grouped GEMM on dense Torch tensors."""
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("a and b must be rank-3 tensors")
    if scale_a.ndim != 3 or scale_b.ndim != 3:
        raise ValueError("scale_a and scale_b must be rank-3 tensors")
    if token_info.ndim != 1:
        raise ValueError("token_info must be a rank-1 tensor")
    tensors = (a, scale_a, b, scale_b, token_info)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("a, scale_a, b, scale_b, and token_info must be CUDA tensors")
    if any(tensor.device != a.device for tensor in tensors[1:]):
        raise ValueError("a, scale_a, b, scale_b, and token_info must be on the same CUDA device")
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise TypeError("a and b must have dtype torch.float8_e4m3fn")
    if scale_a.dtype != torch.float32 or scale_b.dtype != torch.float32:
        raise TypeError("scale_a and scale_b must have dtype torch.float32")
    if token_info.dtype != torch.int32:
        raise TypeError("token_info must have dtype torch.int32")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("a, scale_a, b, scale_b, and token_info must be contiguous")
    if any(tensor.data_ptr() % 16 != 0 for tensor in tensors):
        raise ValueError("a, scale_a, b, scale_b, and token_info must be 16-byte aligned")
    if recipe_a != (1, 128):
        raise ValueError("recipe_a must be (1, 128)")
    if recipe_b != (128, 128):
        raise ValueError("recipe_b must be (128, 128)")

    groups, max_m, gemm_k = a.shape
    groups_b, n, gemm_k_b = b.shape
    if groups_b != groups:
        raise ValueError("a and b must have the same group count")
    if gemm_k_b != gemm_k:
        raise ValueError("a and b must have the same K dimension")
    if token_info.shape != (groups,):
        raise ValueError("token_info must have shape [G]")
    if min(groups, max_m, n, gemm_k) <= 0:
        raise ValueError("G, Mmax, N, and K must be positive")
    _, _, tile_k = get_m_grouped_masked_gemm_fp8_tile()
    if gemm_k % tile_k != 0:
        raise ValueError(f"K must be divisible by {tile_k}")
    if n % recipe_b[0] != 0:
        raise ValueError(f"N must be divisible by {recipe_b[0]}")
    expected_scale_a_shape = (groups, max_m, gemm_k // recipe_a[1])
    expected_scale_b_shape = (groups, n // recipe_b[0], gemm_k // recipe_b[1])
    if scale_a.shape != expected_scale_a_shape:
        raise ValueError(f"scale_a must have shape {expected_scale_a_shape}")
    if scale_b.shape != expected_scale_b_shape:
        raise ValueError(f"scale_b must have shape {expected_scale_b_shape}")
    if block_swizzle_factor <= 0:
        raise ValueError("block_swizzle_factor must be positive")

    capability = torch.cuda.get_device_capability(a.device)
    if capability != (12, 0):
        raise RuntimeError(f"SM120 CUDA device is required, got capability {capability}")

    if out is None:
        out = torch.empty((groups, max_m, n), dtype=torch.bfloat16, device=a.device)
    else:
        if out.ndim != 3:
            raise ValueError("out must be a rank-3 tensor")
        if not out.is_cuda:
            raise ValueError("out must be a CUDA tensor")
        if out.device != a.device:
            raise ValueError("out must be on the same CUDA device as a")
        if out.dtype != torch.bfloat16:
            raise TypeError("out must have dtype torch.bfloat16")
        if out.shape != (groups, max_m, n):
            raise ValueError("out must have shape [G, Mmax, N]")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if out.data_ptr() % 16 != 0:
            raise ValueError("out must be 16-byte aligned")

    with torch.cuda.device(a.device):
        stream = cuda.CUstream(torch.cuda.current_stream(device=a.device).cuda_stream)
        m_grouped_masked_gemm_fp8_sm120(
            from_dlpack(a.detach(), assumed_align=16),
            from_dlpack(scale_a.detach(), assumed_align=16),
            from_dlpack(b.detach(), assumed_align=16),
            from_dlpack(scale_b.detach(), assumed_align=16),
            from_dlpack(out, assumed_align=16),
            from_dlpack(token_info.detach(), assumed_align=16),
            stream,
            bool(use_block_swizzle),
            int(block_swizzle_factor),
            recipe_a,
            recipe_b,
        )
    return out
