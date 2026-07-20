import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute


@cute.jit
def m_grouped_masked_gemm_sm120(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mD: cute.Tensor,
    mMaskedM: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    """Launch SM120 M-grouped masked GEMM on caller-owned CuTe tensors."""
    assert cute.rank(mA) == cute.rank(mB) == cute.rank(mD) == 3
    assert cute.rank(mMaskedM) == 1

    groups, max_m, gemm_k = mA.shape
    groups_b, gemm_n, gemm_k_b = mB.shape
    groups_d, max_m_d, gemm_n_d = mD.shape
    assert groups_b == groups_d == groups
    assert max_m_d == max_m
    assert gemm_k_b == gemm_k
    assert gemm_n_d == gemm_n
    assert mMaskedM.shape == (groups,)
    assert groups > 0 and max_m > 0 and gemm_n > 0 and gemm_k > 0

    dtype = mA.element_type
    assert dtype in (cutlass.Float16, cutlass.BFloat16)
    assert mB.element_type == mD.element_type == dtype
    assert mMaskedM.element_type == cutlass.Int32

    raise NotImplementedError("SM120 M-grouped masked GEMM kernel is not implemented yet")
