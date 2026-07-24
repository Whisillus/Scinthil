from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync

from .group_gemm_base import GroupedGEMMBase
from .grouped_gemm_utils import make_acc_tensor_mn_view
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

    def get_g2s_load_atom(self) -> None:
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
        self.scale_num_bits_per_copy = 32
        self.load_scale_a_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            self.scale_dtype,
            num_bits_per_copy=self.scale_num_bits_per_copy,
        )
        self.load_scale_b_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            self.scale_dtype,
            num_bits_per_copy=self.scale_num_bits_per_copy,
        )

    def get_g2s_load(self) -> None:
        self.get_g2s_load_atom()

        a_copy_elems = self.num_bits_per_copy // self.dtype_a.width
        b_copy_elems = self.num_bits_per_copy // self.dtype_b.width
        self.a_vectors_per_row = self.sA_layout_atom.outer.shape[1] // a_copy_elems
        self.b_vectors_per_row = self.sB_layout_atom.outer.shape[1] // b_copy_elems
        assert self.producer_threads % self.a_vectors_per_row == 0
        assert self.producer_threads % self.b_vectors_per_row == 0

        self.a_rows_per_copy = self.producer_threads // self.a_vectors_per_row
        self.b_rows_per_copy = self.producer_threads // self.b_vectors_per_row
        assert self.tile_m % self.a_rows_per_copy == 0
        assert self.tile_n % self.b_rows_per_copy == 0

        a_thread_layout = cute.make_layout(
            (self.a_rows_per_copy, self.a_vectors_per_row),
            stride=(self.a_vectors_per_row, 1),
        )
        b_thread_layout = cute.make_layout(
            (self.b_rows_per_copy, self.b_vectors_per_row),
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

        scale_a_thread_layout = cute.make_layout(
            (self.scale_a_per_tile,),
            stride=(1,),
        )
        scale_b_thread_layout = cute.make_layout(
            (self.scale_b_per_tile,),
            stride=(1,),
        )
        scale_value_layout = cute.make_layout(
            (1,),
            stride=(1,),
        )
        self.tiled_copy_scale_a = cute.make_tiled_copy_tv(
            self.load_scale_a_atom,
            scale_a_thread_layout,
            scale_value_layout,
        )
        self.tiled_copy_scale_b = cute.make_tiled_copy_tv(
            self.load_scale_b_atom,
            scale_b_thread_layout,
            scale_value_layout,
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

    def get_shared_storage(self):
        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[self.dtype_a, self.sA_size],
                128,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.dtype_b, self.sB_size],
                128,
            ]
            sScaleA: cute.struct.Align[
                cute.struct.MemRange[self.scale_dtype, self.sScaleA_size],
                16,
            ]
            sScaleB: cute.struct.Align[
                cute.struct.MemRange[self.scale_dtype, self.sScaleB_size],
                16,
            ]
            a_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            b_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]

        return SharedStorage

    @cute.jit
    def load(
        self,
        mA: cute.Tensor,
        mScaleA: cute.Tensor,
        mB: cute.Tensor,
        mScaleB: cute.Tensor,
        sA: cute.Tensor,
        sScaleA: cute.Tensor,
        sB: cute.Tensor,
        sScaleB: cute.Tensor,
        tiled_copy_a: cute.TiledCopy,
        tiled_copy_scale_a: cute.TiledCopy,
        tiled_copy_b: cute.TiledCopy,
        tiled_copy_scale_b: cute.TiledCopy,
        tile_m_idx: cutlass.Int32,
        tile_n_idx: cutlass.Int32,
        group_idx: cutlass.Int32,
        valid_m: cutlass.Int32,
        n: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        a_pipeline: pipeline.PipelineCpAsync,
        b_pipeline: pipeline.PipelineCpAsync,
        a_producer_state: pipeline.PipelineState,
        b_producer_state: pipeline.PipelineState,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState]:
        producer_thread_idx = cute.arch.thread_idx()[0] - self.consumer_threads
        local_m = producer_thread_idx // self.a_vectors_per_row
        local_n = producer_thread_idx // self.b_vectors_per_row

        gA = cute.local_tile(
            mA[group_idx, None, None],
            (self.tile_m, self.tile_k),
            (tile_m_idx, None),
        )
        gB = cute.local_tile(
            mB[group_idx, None, None],
            (self.tile_n, self.tile_k),
            (tile_n_idx, None),
        )
        gScaleA = cute.local_tile(
            mScaleA[group_idx, None, None],
            (self.scale_a_per_tile,),
            (tile_m_idx, None),
        )
        scale_b_tile_idx = tile_n_idx // (self.recipe_b[0] // self.tile_n)
        gScaleB = cute.local_tile(
            mScaleB[group_idx, None, None],
            (self.scale_b_per_tile,),
            (scale_b_tile_idx, None),
        )
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
        gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

        a_thr_copy = tiled_copy_a.get_slice(producer_thread_idx)
        tAgA = a_thr_copy.partition_S(gA)
        tAsA = a_thr_copy.partition_D(sA)

        b_thr_copy = tiled_copy_b.get_slice(producer_thread_idx)
        tBgB = b_thr_copy.partition_S(gB)
        tBsB = b_thr_copy.partition_D(sB)

        scale_a_thr_copy = tiled_copy_scale_a.get_slice(producer_thread_idx % self.scale_a_per_tile)
        tSAgScaleA = scale_a_thr_copy.partition_S(gScaleA)
        tSAsScaleA = scale_a_thr_copy.partition_D(sScaleA)

        scale_b_thr_copy = tiled_copy_scale_b.get_slice(0)
        tSBgScaleB = scale_b_thr_copy.partition_S(gScaleB)
        tSBsScaleB = scale_b_thr_copy.partition_D(sScaleB)

        tApA = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tAgA.shape[0][1],
                    cute.size(tAgA, mode=[1]),
                    cute.size(tAgA, mode=[2]),
                ),
                stride=(cute.size(tAgA, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        tBpB = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tBgB.shape[0][1],
                    cute.size(tBgB, mode=[1]),
                    cute.size(tBgB, mode=[2]),
                ),
                stride=(cute.size(tBgB, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        for m_copy_idx in cutlass.range_constexpr(cute.size(tApA, mode=[1])):
            m_idx = tile_m_idx * self.tile_m + local_m + m_copy_idx * self.a_rows_per_copy
            tApA[None, m_copy_idx, None].fill(m_idx < valid_m)
        for n_copy_idx in cutlass.range_constexpr(cute.size(tBpB, mode=[1])):
            n_idx = tile_n_idx * self.tile_n + local_n + n_copy_idx * self.b_rows_per_copy
            tBpB[None, n_copy_idx, None].fill(n_idx < n)

        scale_a_m_idx = tile_m_idx * self.tile_m + producer_thread_idx

        for k_tile_idx in cutlass.range(k_tile_count):
            a_pipeline.producer_acquire(a_producer_state)
            cute.copy(
                tiled_copy_a,
                tAgA[None, None, None, k_tile_idx],
                tAsA[None, None, None, a_producer_state.index],
                pred=tApA,
            )
            if producer_thread_idx < self.scale_a_per_tile and scale_a_m_idx < valid_m:
                cute.copy(
                    tiled_copy_scale_a,
                    tSAgScaleA[None, None, k_tile_idx],
                    tSAsScaleA[None, None, a_producer_state.index],
                )
            a_pipeline.producer_commit(a_producer_state)
            a_producer_state.advance()

            b_pipeline.producer_acquire(b_producer_state)
            cute.copy(
                tiled_copy_b,
                tBgB[None, None, None, k_tile_idx],
                tBsB[None, None, None, b_producer_state.index],
                pred=tBpB,
            )
            if producer_thread_idx < self.scale_b_per_tile:
                cute.copy(
                    tiled_copy_scale_b,
                    tSBgScaleB[None, None, k_tile_idx],
                    tSBsScaleB[None, None, b_producer_state.index],
                )
            b_pipeline.producer_commit(b_producer_state)
            b_producer_state.advance()

        return a_producer_state, b_producer_state

    @cute.jit
    def compute(
        self,
        sA: cute.Tensor,
        sScaleA: cute.Tensor,
        sB: cute.Tensor,
        sScaleB: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tiled_copy_a_s2r: cute.TiledCopy,
        tiled_copy_b_s2r: cute.TiledCopy,
        tile_m_idx: cutlass.Int32,
        valid_m: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        a_pipeline: pipeline.PipelineCpAsync,
        b_pipeline: pipeline.PipelineCpAsync,
        a_consumer_state: pipeline.PipelineState,
        b_consumer_state: pipeline.PipelineState,
    ) -> tuple[cute.Tensor, pipeline.PipelineState, pipeline.PipelineState]:
        consumer_thread_idx = cute.arch.thread_idx()[0]
        thr_mma = tiled_mma.get_slice(consumer_thread_idx)

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])

        a_thr_copy = tiled_copy_a_s2r.get_slice(consumer_thread_idx)
        tAsA = a_thr_copy.partition_S(sA)
        tArA = a_thr_copy.retile(tCrA)

        b_thr_copy = tiled_copy_b_s2r.get_slice(consumer_thread_idx)
        tBsB = b_thr_copy.partition_S(sB)
        tBrB = b_thr_copy.retile(tCrB)

        num_mma_k_blocks = cute.size(tAsA, mode=[2])
        assert num_mma_k_blocks == self.tile_k // self.mma_warp_tile_shape_mnk[2]
        assert num_mma_k_blocks == cute.size(tBsB, mode=[2])

        acc_shape = thr_mma.partition_shape_C((self.tile_m, self.tile_n))
        final_accumulators = cute.make_rmem_tensor(acc_shape, self.dtype_acc)
        temporary_accumulators = cute.make_rmem_tensor(acc_shape, self.dtype_acc)
        final_accumulators.fill(0.0)

        final_accumulators_mn = make_acc_tensor_mn_view(final_accumulators)
        temporary_accumulators_mn = make_acc_tensor_mn_view(temporary_accumulators)
        assert cute.size(final_accumulators_mn, mode=[0]) == 2
        assert cute.size(final_accumulators_mn, mode=[1]) == 4
        lane_m = cute.arch.lane_idx() // 4

        for _ in cutlass.range(k_tile_count):
            a_pipeline.consumer_wait(a_consumer_state)
            b_pipeline.consumer_wait(b_consumer_state)
            temporary_accumulators.fill(0.0)

            for mma_k_block in cutlass.range_constexpr(num_mma_k_blocks):
                cute.copy(
                    tiled_copy_a_s2r,
                    tAsA[None, None, mma_k_block, a_consumer_state.index],
                    tArA[None, None, mma_k_block],
                )
                cute.copy(
                    tiled_copy_b_s2r,
                    tBsB[None, None, mma_k_block, b_consumer_state.index],
                    tBrB[None, None, mma_k_block],
                )
                cute.gemm(
                    tiled_mma,
                    temporary_accumulators,
                    tCrA[None, None, mma_k_block],
                    tCrB[None, None, mma_k_block],
                    temporary_accumulators,
                )

            scale_b = sScaleB[0, b_consumer_state.index]
            scale_a_0 = cutlass.Float32(0.0)
            scale_a_1 = cutlass.Float32(0.0)
            if tile_m_idx * self.tile_m + lane_m < valid_m:
                scale_a_0 = sScaleA[lane_m, a_consumer_state.index]
            if tile_m_idx * self.tile_m + lane_m + 8 < valid_m:
                scale_a_1 = sScaleA[lane_m + 8, a_consumer_state.index]

            a_pipeline.consumer_release(a_consumer_state)
            b_pipeline.consumer_release(b_consumer_state)
            a_consumer_state.advance()
            b_consumer_state.advance()

            scale_a = (scale_a_0, scale_a_1)
            for row in cutlass.range_constexpr(cute.size(final_accumulators_mn, mode=[0])):
                final_row = final_accumulators_mn[row, None]
                temporary_row = temporary_accumulators_mn[row, None]
                final_row.store(final_row.load() + temporary_row.load() * scale_a[row] * scale_b)

        return final_accumulators, a_consumer_state, b_consumer_state

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mScaleA: cute.Tensor,
        mB: cute.Tensor,
        mScaleB: cute.Tensor,
        mD: cute.Tensor,
        mInfo: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sScaleA_layout: cute.Layout,
        sB_layout: cute.ComposedLayout,
        sScaleB_layout: cute.Layout,
        tiled_copy_a: cute.TiledCopy,
        tiled_copy_scale_a: cute.TiledCopy,
        tiled_copy_b: cute.TiledCopy,
        tiled_copy_scale_b: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tiled_copy_a_s2r: cute.TiledCopy,
        tiled_copy_b_s2r: cute.TiledCopy,
        SharedStorage: cutlass.Constexpr,
    ) -> None:
        groups, max_m, gemm_k = mA.shape
        _, n, _ = mB.shape
        num_tile_m = cute.ceil_div(max_m, self.tile_m)
        num_tile_n = cute.ceil_div(n, self.tile_n)
        k_tile_count = gemm_k // self.tile_k

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        is_consumer = warp_idx < self.num_consumer_warps
        is_producer = warp_idx >= self.num_consumer_warps
        is_producer &= warp_idx < self.num_consumer_warps + self.num_producer_warps

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = storage.sA.get_tensor(sA_layout.outer, swizzle=sA_layout.inner)
        sB = storage.sB.get_tensor(sB_layout.outer, swizzle=sB_layout.inner)
        sScaleA = storage.sScaleA.get_tensor(sScaleA_layout)
        sScaleB = storage.sScaleB.get_tensor(sScaleB_layout)

        producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.producer_threads,
        )
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.consumer_threads,
        )
        a_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.a_mbar.data_ptr(),
            num_stages=self.stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            defer_sync=True,
        )
        b_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.b_mbar.data_ptr(),
            num_stages=self.stages,
            producer_group=producer_group,
            consumer_group=consumer_group,
            defer_sync=True,
        )
        pipeline.pipeline_init_arrive()
        pipeline.pipeline_init_wait()

        block_idx_x, _, _ = cute.arch.block_idx()
        grid_dim_x, _, _ = cute.arch.grid_dim()
        scheduler = PersistentTileScheduler.create(
            self.tile_m,
            self.use_block_swizzle,
            self.block_swizzle_factor,
            num_tile_m,
            num_tile_n,
            groups,
            mInfo,
            block_idx_x,
            grid_dim_x,
        )
        a_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            self.stages,
        )
        b_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            self.stages,
        )
        a_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.stages,
        )
        b_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.stages,
        )
        work_tile = scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            tile_m_idx, tile_n_idx, group_idx = work_tile.tile_idx
            valid_m = mInfo[group_idx]
            if is_producer:
                a_producer_state, b_producer_state = self.load(
                    mA,
                    mScaleA,
                    mB,
                    mScaleB,
                    sA,
                    sScaleA,
                    sB,
                    sScaleB,
                    tiled_copy_a,
                    tiled_copy_scale_a,
                    tiled_copy_b,
                    tiled_copy_scale_b,
                    tile_m_idx,
                    tile_n_idx,
                    group_idx,
                    valid_m,
                    n,
                    k_tile_count,
                    a_pipeline,
                    b_pipeline,
                    a_producer_state,
                    b_producer_state,
                )
            if is_consumer:
                _accumulators, a_consumer_state, b_consumer_state = self.compute(
                    sA,
                    sScaleA,
                    sB,
                    sScaleB,
                    tiled_mma,
                    tiled_copy_a_s2r,
                    tiled_copy_b_s2r,
                    tile_m_idx,
                    valid_m,
                    k_tile_count,
                    a_pipeline,
                    b_pipeline,
                    a_consumer_state,
                    b_consumer_state,
                )
            work_tile = scheduler.advance_to_next_work()

        if is_producer:
            a_pipeline.producer_tail(a_producer_state)
            b_pipeline.producer_tail(b_producer_state)

        del mD

    def __call__(
        self,
        mA: cute.Tensor,
        mScaleA: cute.Tensor,
        mB: cute.Tensor,
        mScaleB: cute.Tensor,
        mD: cute.Tensor,
        mInfo: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        self.get_warpspecialize_config()
        self.get_mma_atom()
        self.get_tiled_mma()
        self.get_smem_layout()
        self.get_g2s_load()
        self.get_ab_s2r()
        self.get_shared_storage()
        groups, max_m, _ = mA.shape
        _, n, _ = mB.shape
        PersistentTileScheduler.get_grid_shape(
            cute.ceil_div(max_m, self.tile_m),
            cute.ceil_div(n, self.tile_n),
            groups,
            self.max_persistent_ctas,
        )
        del mScaleA, mScaleB, mD, mInfo, stream
        raise NotImplementedError("FP8 BF16 epilogue is not implemented")
