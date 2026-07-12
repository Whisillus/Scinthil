import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from .fa_fwd_base import FlashAttentionForwardBase
from .tile_scheduler import SingleTileScheduler


class FlashAttentionForwardM16N8K16SM120(FlashAttentionForwardBase):
    """Warp-specialized FlashAttention forward kernel configuration."""

    def can_implement(self) -> bool:
        if self.dtype not in (cutlass.Float16, cutlass.BFloat16):
            return False
        if self.acc_dtype not in (cutlass.Float16, cutlass.Float32):
            return False
        if self.dtype == cutlass.BFloat16 and self.acc_dtype != cutlass.Float32:
            return False
        if self.headdim_qk % 64 != 0 or self.headdim_v % 64 != 0:
            return False
        if self.tile_m % 16 != 0 or self.tile_n % 16 != 0:
            return False
        if self.stage_k <= 0 or self.stage_v <= 0:
            return False
        if self.num_producer <= 0 or self.num_consumer <= 0:
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

    def get_qk_pv_mma_atom(self) -> None:
        mma_inst_shape_mnk = (16, 8, 16)
        self.mma_atom_qk = cute.make_mma_atom(
            cute.nvgpu.warp.MmaF16BF16Op(
                self.dtype,
                self.acc_dtype,
                mma_inst_shape_mnk,
            )
        )
        self.mma_atom_pv = cute.make_mma_atom(
            cute.nvgpu.warp.MmaF16BF16Op(
                self.dtype,
                self.acc_dtype,
                mma_inst_shape_mnk,
            )
        )

    def get_qk_pv_tiled_mma(self) -> None:
        self.get_qk_pv_mma_atom()
        atom_shape_mnk = (self.num_consumer, 1, 1)
        atom_stride_mnk = (1, self.num_consumer, self.num_consumer)
        atom_layout_mnk = cute.make_layout(
            atom_shape_mnk,
            stride=atom_stride_mnk,
        )
        # Extend the native N=8 atom to N=16 so each warp can load the two
        # adjacent B fragments together with ldmatrix.x4 and reuse A for both MMAs.
        permutation_mnk = (self.num_consumer * 16, 16, 16)
        self.tiled_mma_qk = cute.make_tiled_mma(
            self.mma_atom_qk,
            atom_layout_mnk,
            permutation_mnk=permutation_mnk,
        )
        self.tiled_mma_pv = cute.make_tiled_mma(
            self.mma_atom_pv,
            atom_layout_mnk,
            permutation_mnk=permutation_mnk,
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        softmax_scale: cutlass.Float32,
    ) -> None:
        self.get_smem_layout()
        self.get_qk_pv_tiled_mma()
        num_batch, seqlen_q, num_head_q, _ = mQ.shape
        num_tile_m = cute.ceil_div(seqlen_q, self.tile_m)
        grid = SingleTileScheduler.get_grid_shape(
            num_tile_m,
            num_head_q,
            num_batch,
        )
        self.kernel(mQ, mK, mV, mO, softmax_scale).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
        )

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


def check_compile(fa_fwd: FlashAttentionForwardM16N8K16SM120):
    if not fa_fwd.can_implement():
        raise ValueError("FlashAttention configuration cannot be implemented on SM120")

    make_tensor = cute.runtime.make_fake_compact_tensor
    tensor_options = {
        "stride_order": (3, 2, 1, 0),
        "assumed_align": 16,
    }
    mQ = make_tensor(
        fa_fwd.dtype,
        (1, fa_fwd.tile_m, 1, fa_fwd.headdim_qk),
        **tensor_options,
    )
    mK = make_tensor(
        fa_fwd.dtype,
        (1, fa_fwd.tile_n, 1, fa_fwd.headdim_qk),
        **tensor_options,
    )
    mV = make_tensor(
        fa_fwd.dtype,
        (1, fa_fwd.tile_n, 1, fa_fwd.headdim_v),
        **tensor_options,
    )
    mO = make_tensor(
        fa_fwd.dtype,
        (1, fa_fwd.tile_m, 1, fa_fwd.headdim_v),
        **tensor_options,
    )
    compile_options = (cute.GPUArch("sm_120a"), cute.EnableTVMFFI)
    return cute.compile[compile_options](
        fa_fwd,
        mQ,
        mK,
        mV,
        mO,
        cutlass.Float32(1.0),
    )
