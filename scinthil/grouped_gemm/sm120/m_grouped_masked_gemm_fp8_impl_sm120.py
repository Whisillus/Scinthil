from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync

from .group_gemm_base import GroupedGEMMBase
from .scheduler import PersistentTileScheduler


class MGroupedMaskedGEMMFP8SM120(GroupedGEMMBase):
    def __init__(
        self,
        *,
        dtype_a: type[Any],
        dtype_b: type[Any],
        dtype_d: type[Any],
        scale_dtype: type[Any],
        recipe_a: tuple[int, int],
        recipe_b: tuple[int, int],
        tile_m: int,
        tile_n: int,
        tile_k: int,
        use_block_swizzle: bool,
        block_swizzle_factor: int = 8,
    ) -> None:
        super().__init__(
            dtype_a=dtype_a,
            dtype_b=dtype_b,
            dtype_d=dtype_d,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            use_block_swizzle=use_block_swizzle,
            block_swizzle_factor=block_swizzle_factor,
        )
        self.scale_dtype = scale_dtype
        self.recipe_a = recipe_a
        self.recipe_b = recipe_b
        assert self.dtype_a == self.dtype_b == cutlass.Float8E4M3FN
        assert self.scale_dtype == cutlass.Float32
        assert self.dtype_d == cutlass.BFloat16
        assert self.recipe_a == (1, 128)
        assert self.recipe_b == (128, 128)
        assert self.tile_k == self.recipe_a[1] == self.recipe_b[1]
        assert self.recipe_b[0] % self.tile_n == 0
        self.scale_a_per_tile = self.tile_m // self.recipe_a[0]
        self.scale_b_per_tile = 1

    def get_warpspecialize_config(self) -> None:
        self.mma_inst_shape_mnk = (16, 8, 32)
        self.mma_warp_tile_shape_mnk = (16, 16, 32)
        assert self.tile_m % self.mma_warp_tile_shape_mnk[0] == 0
        assert self.tile_n % self.mma_warp_tile_shape_mnk[1] == 0
        assert self.tile_k % self.mma_warp_tile_shape_mnk[2] == 0

        self.num_consumer_warps = 1
        self.num_producer_warps = 1
        self.consumer_threads = self.num_consumer_warps * cute.arch.WARP_SIZE
        self.producer_threads = self.num_producer_warps * cute.arch.WARP_SIZE
        self.threads_per_cta = self.consumer_threads + self.producer_threads

    def get_mma_atom(self) -> None:
        assert self.dtype_a == self.dtype_b == cutlass.Float8E4M3FN
        assert self.dtype_acc == cutlass.Float32
        self.mma_atom = cute.make_mma_atom(
            cute.nvgpu.warp.MmaFP8Op(
                self.dtype_a,
                self.dtype_acc,
                self.mma_inst_shape_mnk,
            )
        )

    def get_tiled_mma(self) -> None:
        atom_layout_mnk = cute.make_layout(
            (self.num_consumer_warps, 1, 1),
            stride=(1, self.num_consumer_warps, self.num_consumer_warps),
        )
        self.tiled_mma = cute.make_tiled_mma(
            self.mma_atom,
            atom_layout_mnk,
            permutation_mnk=self.mma_warp_tile_shape_mnk,
        )

    def get_smem_layout_atom(self) -> None:
        layout_atom_outer = cute.make_layout(
            (8, self.tile_k),
            stride=(self.tile_k, 1),
        )
        self.sA_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            layout_atom_outer,
        )
        self.sB_layout_atom = self.sA_layout_atom

    def get_smem_layout(self) -> None:
        self.get_smem_layout_atom()
        self.sA_layout = cute.tile_to_shape(
            self.sA_layout_atom,
            (self.tile_m, self.tile_k, self.stages),
            (0, 1, 2),
        )
        self.sB_layout = cute.tile_to_shape(
            self.sB_layout_atom,
            (self.tile_n, self.tile_k, self.stages),
            (0, 1, 2),
        )
        self.sScaleA_layout = cute.make_layout(
            (self.scale_a_per_tile, self.stages),
            stride=(1, self.scale_a_per_tile),
        )
        self.sScaleB_layout = cute.make_layout(
            (self.scale_b_per_tile, self.stages),
            stride=(1, 1),
        )
        self.sA_size = cute.cosize(self.sA_layout)
        self.sB_size = cute.cosize(self.sB_layout)
        self.sScaleA_size = cute.cosize(self.sScaleA_layout)
        self.sScaleB_size = cute.cosize(self.sScaleB_layout)

    def get_ab_load_atom(self) -> None:
        self.num_bits_per_copy = 128
        self.load_a_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype_a,
            num_bits_per_copy=self.num_bits_per_copy,
        )
        self.load_b_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype_b,
            num_bits_per_copy=self.num_bits_per_copy,
        )

    def get_ab_load(self) -> None:
        self.get_ab_load_atom()

        a_copy_elems = self.num_bits_per_copy // self.dtype_a.width
        b_copy_elems = self.num_bits_per_copy // self.dtype_b.width
        self.a_vectors_per_row = self.sA_layout_atom.outer.shape[1] // a_copy_elems
        self.b_vectors_per_row = self.sB_layout_atom.outer.shape[1] // b_copy_elems
        assert self.producer_threads % self.a_vectors_per_row == 0
        assert self.producer_threads % self.b_vectors_per_row == 0

        a_rows_per_copy = self.producer_threads // self.a_vectors_per_row
        b_rows_per_copy = self.producer_threads // self.b_vectors_per_row
        assert self.tile_m % a_rows_per_copy == 0
        assert self.tile_n % b_rows_per_copy == 0

        a_thread_layout = cute.make_layout(
            (a_rows_per_copy, self.a_vectors_per_row),
            stride=(self.a_vectors_per_row, 1),
        )
        b_thread_layout = cute.make_layout(
            (b_rows_per_copy, self.b_vectors_per_row),
            stride=(self.b_vectors_per_row, 1),
        )
        a_value_layout = cute.make_layout(
            (1, a_copy_elems),
            stride=(a_copy_elems, 1),
        )
        b_value_layout = cute.make_layout(
            (1, b_copy_elems),
            stride=(b_copy_elems, 1),
        )
        self.tiled_copy_a = cute.make_tiled_copy_tv(
            self.load_a_atom,
            a_thread_layout,
            a_value_layout,
        )
        self.tiled_copy_b = cute.make_tiled_copy_tv(
            self.load_b_atom,
            b_thread_layout,
            b_value_layout,
        )

    def get_ab_s2r_atom(self) -> None:
        self.s2r_a_atom = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                transpose=False,
                num_matrices=4,
            ),
            self.dtype_a,
        )
        self.s2r_b_atom = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(
                transpose=False,
                num_matrices=4,
            ),
            self.dtype_b,
        )

    def get_ab_s2r(self) -> None:
        self.get_ab_s2r_atom()
        self.tiled_copy_a_s2r = cute.make_tiled_copy_A(self.s2r_a_atom, self.tiled_mma)
        self.tiled_copy_b_s2r = cute.make_tiled_copy_B(self.s2r_b_atom, self.tiled_mma)

    def __call__(
        self,
        mA: cute.Tensor,
        scale_a: cute.Tensor,
        mB: cute.Tensor,
        scale_b: cute.Tensor,
        mD: cute.Tensor,
        mInfo: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        self.get_warpspecialize_config()
        self.get_mma_atom()
        self.get_tiled_mma()
        self.get_smem_layout()
        self.get_ab_load()
        self.get_ab_s2r()
        groups, max_m, _ = mA.shape
        _, n, _ = mB.shape
        PersistentTileScheduler.get_grid_shape(
            cute.ceil_div(max_m, self.tile_m),
            cute.ceil_div(n, self.tile_n),
            groups,
            self.max_persistent_ctas,
        )
        del scale_a, scale_b, mD, mInfo, stream
        raise NotImplementedError("FP8 scale loading and scaled MMA mainloop are not implemented")
