from dataclasses import dataclass

import cutlass
import cutlass.cute as cute


@dataclass(frozen=True)
class BlockInfo:
    tile_m: int
    tile_n: int
    is_causal: bool

    @cute.jit
    def get_n_block_min_max(
        self,
        seqlen_q: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        m_block: cutlass.Int32,
    ) -> tuple[cutlass.Int32, cutlass.Int32]:
        n_block_min = cutlass.Int32(0)
        n_block_max = cute.ceil_div(seqlen_k, self.tile_n)

        if cutlass.const_expr(self.is_causal):
            m_idx_max = (m_block + 1) * self.tile_m
            n_idx_right = m_idx_max + seqlen_k - seqlen_q
            n_block_max = cutlass.min(n_block_max, cute.ceil_div(n_idx_right, self.tile_n))
            n_block_max = cutlass.max(n_block_max, n_block_min)

        return n_block_min, n_block_max
