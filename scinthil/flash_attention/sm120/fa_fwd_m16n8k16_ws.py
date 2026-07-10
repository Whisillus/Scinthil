import cutlass
import cutlass.cute as cute

from .fa_fwd_base import FlashAttentionForwardBase


class FlashAttentionForwardM16N8K16SM120(FlashAttentionForwardBase):
    """Warp-specialized FlashAttention forward kernel configuration."""

    def get_smem_layout_atom(self):
        raise NotImplementedError

    def get_smem_layout(self):
        raise NotImplementedError

    def get_qk_pv_mma_atom(self):
        raise NotImplementedError

    def get_qk_pv_tiled_mma(self):
        raise NotImplementedError

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        softmax_scale: cutlass.Float32,
    ) -> None:
        raise NotImplementedError
