from dataclasses import dataclass

import cutlass
import cutlass.cute as cute

from .fa_utils import make_acc_tensor_mn_view


@dataclass(frozen=True)
class FlashAttentionMaskSM120:
    tile_m: int
    tile_n: int
    is_causal: bool
    tScS: cute.Tensor

    @staticmethod
    def create(
        qk_thr_mma: cute.ThrMma,
        tile_m: int,
        tile_n: int,
        is_causal: bool,
    ) -> "FlashAttentionMaskSM120":
        cS = cute.make_identity_tensor((tile_m, tile_n))
        tScS = qk_thr_mma.partition_C(cS)

        return FlashAttentionMaskSM120(tile_m, tile_n, is_causal, tScS)

    @cute.jit
    def apply(
        self,
        acc_s: cute.Tensor,
        tile_m_idx: cutlass.Int32,
        tile_n_idx: cutlass.Int32,
        seqlen_q: cutlass.Int32,
        seqlen_k: cutlass.Int32,
    ) -> None:
        acc_s_mn = make_acc_tensor_mn_view(acc_s)
        tScS_mn = make_acc_tensor_mn_view(self.tScS)

        tile_n_offset = tile_n_idx * self.tile_n
        if cutlass.const_expr(self.is_causal):
            for row in cutlass.range_constexpr(cute.size(acc_s_mn.shape[0])):
                m_coord, _ = tScS_mn[row, 0]
                m_idx = tile_m_idx * self.tile_m + m_coord
                max_n_coord = m_idx + seqlen_k - seqlen_q - tile_n_offset

                for col in cutlass.range_constexpr(cute.size(acc_s_mn.shape[1])):
                    _, n_coord = tScS_mn[0, col]
                    if n_coord > max_n_coord:
                        acc_s_mn[row, col] = -cutlass.Float32.inf
        else:
            col_limit = seqlen_k - tile_n_offset
            for col in cutlass.range_constexpr(cute.size(acc_s_mn.shape[1])):
                _, n_coord = tScS_mn[0, col]
                if n_coord >= col_limit:
                    for row in cutlass.range_constexpr(cute.size(acc_s_mn.shape[0])):
                        acc_s_mn[row, col] = -cutlass.Float32.inf
