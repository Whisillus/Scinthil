from typing import Any

import cuda.bindings.driver as cuda
import cutlass.cute as cute

from .scheduler import PersistentTileScheduler


class MGroupedMaskedGemmSM120:
    def __init__(
        self,
        *,
        dtype: type[Any],
        tile_m: int,
        tile_n: int,
        use_block_swizzle: bool,
        num_1d_blocks_per_group: int = 8,
    ) -> None:
        self.dtype = dtype
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.use_block_swizzle = use_block_swizzle
        self.num_1d_blocks_per_group = num_1d_blocks_per_group
        self.threads_per_cta = 128
        self.max_persistent_ctas = 128

    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        mInfo: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        groups, max_m, _ = mA.shape
        _, gemm_n, _ = mB.shape
        num_tile_m = cute.ceil_div(max_m, self.tile_m)
        num_tile_n = cute.ceil_div(gemm_n, self.tile_n)
        grid = PersistentTileScheduler.get_grid_shape(
            num_tile_m,
            num_tile_n,
            groups,
            self.max_persistent_ctas,
        )
        self.kernel(mA, mB, mD, mInfo).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        mInfo: cute.Tensor,
    ) -> None:
        groups, max_m, _ = mA.shape
        _, gemm_n, _ = mB.shape
        num_tile_m = cute.ceil_div(max_m, self.tile_m)
        num_tile_n = cute.ceil_div(gemm_n, self.tile_n)

        block_idx_x, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()
        scheduler = PersistentTileScheduler.create(
            self.tile_m,
            self.use_block_swizzle,
            self.num_1d_blocks_per_group,
            num_tile_m,
            num_tile_n,
            groups,
            mInfo,
            block_idx_x,
            grid_dim_x,
        )
        work_tile = scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            _tile_m_idx, _tile_n_idx, _group_idx = work_tile.tile_idx
            work_tile = scheduler.advance_to_next_work()
