import torch


def m_grouped_masked_gemm_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("a and b must be rank-3 tensors")
    if a.shape[0] != b.shape[0] or a.shape[2] != b.shape[2]:
        raise ValueError("a and b must have compatible grouped GEMM shapes")
    return torch.einsum("gmk,gnk->gmn", a.float(), b.float()).to(a.dtype)
