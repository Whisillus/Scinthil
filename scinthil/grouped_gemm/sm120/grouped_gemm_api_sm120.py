import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from .m_grouped_masked_gemm_impl_sm120 import MGroupedMaskedGEMMSM120


def get_m_grouped_masked_gemm_tile() -> tuple[int, int, int]:
    """Select the initial CTA tile shape for SM120 M-grouped masked GEMM."""
    tile_m = 16
    tile_n = 16
    tile_k = 16
    return tile_m, tile_n, tile_k


@cute.jit
def m_grouped_masked_gemm_sm120(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mD: cute.Tensor,
    mInfo: cute.Tensor,
    use_block_swizzle: cutlass.Constexpr,
    block_swizzle_factor: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    """Launch SM120 M-grouped masked GEMM on caller-owned CuTe tensors."""
    assert cute.rank(mA) == cute.rank(mB) == cute.rank(mD) == 3
    assert cute.rank(mInfo) == 1

    groups, max_m, gemm_k = mA.shape
    groups_b, n, gemm_k_b = mB.shape
    groups_d, max_m_d, n_d = mD.shape
    assert groups_b == groups_d == groups
    assert max_m_d == max_m
    assert gemm_k_b == gemm_k
    assert n_d == n
    assert mInfo.shape == (groups,)
    assert groups > 0 and max_m > 0 and n > 0 and gemm_k > 0

    dtype = mA.element_type
    assert dtype in (cutlass.Float16, cutlass.BFloat16)
    assert mB.element_type == mD.element_type == dtype
    assert mInfo.element_type == cutlass.Int32
    assert block_swizzle_factor > 0

    tile_m, tile_n, tile_k = get_m_grouped_masked_gemm_tile()
    assert gemm_k % tile_k == 0
    assert mA.stride[2] == 1 and mB.stride[2] == 1 and mD.stride[2] == 1
    gemm = MGroupedMaskedGEMMSM120(
        dtype=dtype,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        use_block_swizzle=use_block_swizzle,
        block_swizzle_factor=block_swizzle_factor,
    )
    gemm(mA, mB, mD, mInfo, stream)
