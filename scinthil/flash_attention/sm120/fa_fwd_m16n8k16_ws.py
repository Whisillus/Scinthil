import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from .fa_fwd_base import FlashAttentionForwardBase


class FlashAttentionForwardM16N8K16SM120(FlashAttentionForwardBase):
    """Warp-specialized FlashAttention forward kernel configuration."""

    def can_implement(self) -> bool:
        if self.dtype not in (cutlass.Float16, cutlass.BFloat16):
            return False
        if self.headdim_qk % 64 != 0 or self.headdim_v % 64 != 0:
            return False
        if self.tile_m % 16 != 0 or self.tile_n % 16 != 0:
            return False
        if self.stage_k <= 0 or self.stage_v <= 0:
            return False

        smem_usage_q = self.tile_m * self.headdim_qk * self.dtype_byte
        smem_usage_k = self.tile_n * self.headdim_qk * self.stage_k * self.dtype_byte
        smem_usage_v = self.tile_n * self.headdim_v * self.stage_v * self.dtype_byte
        # O reuses Q's shared-memory storage during the epilogue.
        smem_usage = smem_usage_q + smem_usage_k + smem_usage_v

        return smem_usage <= utils.get_smem_capacity_in_bytes("sm_120")

    def get_smem_layout_atom(self) -> None:
        # FP16/BF16 elements are 2 bytes and supported head dimensions are
        # multiples of 64, so every row supports this fixed 128-byte swizzle.
        self.sQ_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3),
            0,
            cute.make_ordered_layout((8, 64), order=(1, 0)),
        )
        self.sK_layout_atom = self.sQ_layout_atom
        self.sV_layout_atom = self.sQ_layout_atom
        self.sO_layout_atom = self.sQ_layout_atom

    def get_smem_layout(self) -> None:
        self.get_smem_layout_atom()
        self.sQ_layout = cute.tile_to_shape(
            self.sQ_layout_atom,
            (self.tile_m, self.headdim_qk),
            (0, 1),
        )
        self.sK_layout = cute.tile_to_shape(
            self.sK_layout_atom,
            (self.tile_n, self.headdim_qk, self.stage_k),
            (0, 1, 2),
        )
        # K and V have independent stage counts for the later WS pipelines.
        self.sV_layout = cute.tile_to_shape(
            self.sV_layout_atom,
            (self.tile_n, self.headdim_v, self.stage_v),
            (0, 1, 2),
        )
        self.sO_layout = cute.tile_to_shape(
            self.sO_layout_atom,
            (self.tile_m, self.headdim_v),
            (0, 1),
        )

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
