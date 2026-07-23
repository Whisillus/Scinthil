import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from .m_grouped_masked_gemm_fp8_impl_sm120 import MGroupedMaskedGEMMFP8SM120
from .m_grouped_masked_gemm_impl_sm120 import MGroupedMaskedGEMMSM120


def get_m_grouped_masked_gemm_tile() -> tuple[int, int, int]:
    """Select the initial CTA tile shape for SM120 M-grouped masked GEMM."""
    tile_m = 16
    tile_n = 16
    tile_k = 16
    return tile_m, tile_n, tile_k


def get_m_grouped_masked_gemm_fp8_tile() -> tuple[int, int, int]:
    """Select the correctness-first CTA tile for SM120 FP8 grouped GEMM."""
    return 16, 16, 128


@cute.jit
def m_grouped_masked_gemm_sm120(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mD: cute.Tensor,
    mInfo: cute.Tensor,
    stream: cuda.CUstream,
    use_block_swizzle: cutlass.Constexpr = True,
    block_swizzle_factor: cutlass.Constexpr = 8,
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

    dtype_a = mA.element_type
    dtype_b = mB.element_type
    dtype_d = mD.element_type
    assert dtype_a in (cutlass.Float16, cutlass.BFloat16)
    assert dtype_b == dtype_d == dtype_a
    assert mInfo.element_type == cutlass.Int32
    assert block_swizzle_factor > 0

    tile_m, tile_n, tile_k = get_m_grouped_masked_gemm_tile()
    assert gemm_k % tile_k == 0
    assert mA.stride[2] == 1 and mB.stride[2] == 1 and mD.stride[2] == 1
    gemm = MGroupedMaskedGEMMSM120(
        dtype_a=dtype_a,
        dtype_b=dtype_b,
        dtype_d=dtype_d,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        use_block_swizzle=use_block_swizzle,
        block_swizzle_factor=block_swizzle_factor,
    )
    gemm(mA, mB, mD, mInfo, stream)


@cute.jit
def m_grouped_masked_gemm_fp8_sm120(
    mA: cute.Tensor,
    mScaleA: cute.Tensor,
    mB: cute.Tensor,
    mScaleB: cute.Tensor,
    mD: cute.Tensor,
    mInfo: cute.Tensor,
    stream: cuda.CUstream,
    use_block_swizzle: cutlass.Constexpr = True,
    block_swizzle_factor: cutlass.Constexpr = 8,
    recipe_a: cutlass.Constexpr = (1, 128),
    recipe_b: cutlass.Constexpr = (128, 128),
) -> None:
    """Launch groupwise-scaled SM120 FP8 grouped GEMM on caller-owned tensors."""
    assert cute.rank(mA) == cute.rank(mB) == cute.rank(mD) == 3
    assert cute.rank(mScaleA) == cute.rank(mScaleB) == 3
    assert cute.rank(mInfo) == 1
    assert recipe_a == (1, 128)
    assert recipe_b == (128, 128)

    groups, max_m, gemm_k = mA.shape
    groups_b, n, gemm_k_b = mB.shape
    groups_d, max_m_d, n_d = mD.shape
    assert groups_b == groups_d == groups
    assert max_m_d == max_m
    assert gemm_k_b == gemm_k
    assert n_d == n
    assert mInfo.shape == (groups,)
    assert groups > 0 and max_m > 0 and n > 0 and gemm_k > 0

    assert mA.element_type == mB.element_type == cutlass.Float8E4M3FN
    assert mScaleA.element_type == mScaleB.element_type == cutlass.Float32
    assert mD.element_type == cutlass.BFloat16
    assert mInfo.element_type == cutlass.Int32
    assert block_swizzle_factor > 0

    tile_m, tile_n, tile_k = get_m_grouped_masked_gemm_fp8_tile()
    assert gemm_k % tile_k == 0
    assert n % recipe_b[0] == 0
    assert mScaleA.shape == (groups, max_m, gemm_k // recipe_a[1])
    assert mScaleB.shape == (groups, n // recipe_b[0], gemm_k // recipe_b[1])
    assert mA.stride[2] == 1 and mB.stride[2] == 1 and mD.stride[2] == 1
    assert mScaleA.stride[2] == 1 and mScaleB.stride[2] == 1

    gemm = MGroupedMaskedGEMMFP8SM120(
        dtype_a=mA.element_type,
        dtype_b=mB.element_type,
        dtype_d=mD.element_type,
        scale_dtype=mScaleA.element_type,
        recipe_a=recipe_a,
        recipe_b=recipe_b,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        use_block_swizzle=use_block_swizzle,
        block_swizzle_factor=block_swizzle_factor,
    )
    gemm(mA, mScaleA, mB, mScaleB, mD, mInfo, stream)
