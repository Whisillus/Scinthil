import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

from .fa_fwd_ws_sm120 import FlashAttentionForwardM16N8K16SM120


def get_fa_tile_stage(headdim_qk: int, headdim_v: int) -> tuple[int, int, int, int]:
    """Select CTA tile sizes and K/V pipeline stages."""
    tile_m = 64
    tile_n = 16
    stage_k = 1
    stage_v = 1
    return tile_m, tile_n, stage_k, stage_v


@cute.jit
def flash_attention_bshd_sm120(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mO: cute.Tensor,
    mLSE: cute.Tensor,
    max_seqlen_q: cutlass.Int32,
    max_seqlen_kv: cutlass.Int32,
    softmax_scale: cutlass.Float32,
    is_causal: cutlass.Constexpr,
    stream: cuda.CUstream,
) -> None:
    """Launch SM120 FlashAttention on caller-owned BSHD CuTe tensors."""
    assert cute.rank(mQ) == cute.rank(mK) == cute.rank(mV) == cute.rank(mO) == 4
    assert cute.rank(mLSE) == 3

    batch, _, head_q, headdim_qk = mQ.shape
    batch_k, _, head_kv, headdim_k = mK.shape
    batch_v, _, head_v, headdim_v = mV.shape
    assert batch_k == batch_v == batch
    assert head_v == head_kv
    assert headdim_k == headdim_qk
    assert mO.shape == (batch, mQ.shape[1], head_q, headdim_v)
    assert mLSE.shape == (batch, head_q, mQ.shape[1])

    dtype = mQ.element_type
    assert dtype in (cutlass.Float16, cutlass.BFloat16)
    assert mK.element_type == mV.element_type == mO.element_type == dtype
    assert mLSE.element_type == cutlass.Float32

    tile_m, tile_n, stage_k, stage_v = get_fa_tile_stage(headdim_qk, headdim_v)
    fa_fwd = FlashAttentionForwardM16N8K16SM120(
        dtype=dtype,
        head_q=head_q,
        head_kv=head_kv,
        headdim_qk=headdim_qk,
        headdim_v=headdim_v,
        tile_m=tile_m,
        tile_n=tile_n,
        stage_k=stage_k,
        stage_v=stage_v,
        is_causal=is_causal,
    )
    assert fa_fwd.can_implement(), "FlashAttention configuration cannot be implemented by the SM120 kernel"
    fa_fwd(
        mQ,
        mK,
        mV,
        mO,
        mLSE,
        softmax_scale,
        stream,
    )
