import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync

from .group_gemm_base import GroupedGEMMBase
from .grouped_gemm_utils import make_acc_tensor_mn_view
from .scheduler import PersistentTileScheduler


class MGroupedMaskedGEMMSM120(GroupedGEMMBase):
    def __init__(
        self,
        *,
        dtype_a,
        dtype_b,
        dtype_d,
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

    def get_warpspecialize_config(self) -> None:
        self.mma_inst_shape_mnk = (16, 8, 16)
        self.mma_warp_tile_shape_mnk = (16, 16, 16)
        assert self.tile_m % self.mma_warp_tile_shape_mnk[0] == 0
        assert self.tile_n % self.mma_warp_tile_shape_mnk[1] == 0
        assert self.tile_k % self.mma_warp_tile_shape_mnk[2] == 0

        self.num_consumer_warps = 1
        self.num_producer_warps = 1
        self.consumer_threads = self.num_consumer_warps * cute.arch.WARP_SIZE
        self.producer_threads = self.num_producer_warps * cute.arch.WARP_SIZE
        self.threads_per_cta = self.consumer_threads + self.producer_threads

    def get_mma_atom(self) -> None:
        assert self.dtype_a == self.dtype_b
        self.mma_atom = cute.make_mma_atom(
            cute.nvgpu.warp.MmaF16BF16Op(
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
            cute.make_swizzle(1, 3, 3),
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
        self.sA_size = cute.cosize(self.sA_layout)
        self.sB_size = cute.cosize(self.sB_layout)

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

    def get_g2s_load(self) -> None:
        self.get_g2s_load_atom()

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

    def get_s2r_atom(self) -> None:
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

    def get_s2r_load(self) -> None:
        self.get_s2r_atom()
        self.tiled_copy_a_s2r = cute.make_tiled_copy_A(self.s2r_a_atom, self.tiled_mma)
        self.tiled_copy_b_s2r = cute.make_tiled_copy_B(self.s2r_b_atom, self.tiled_mma)

    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        mInfo: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        self.get_warpspecialize_config()
        self.get_mma_atom()
        self.get_tiled_mma()
        self.get_smem_layout()
        self.get_g2s_load()
        self.get_s2r_load()

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
            a_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            b_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]

        groups, max_m, _ = mA.shape
        _, n, _ = mB.shape
        num_tile_m = cute.ceil_div(max_m, self.tile_m)
        num_tile_n = cute.ceil_div(n, self.tile_n)
        grid = PersistentTileScheduler.get_grid_shape(
            num_tile_m,
            num_tile_n,
            groups,
            self.max_persistent_ctas,
        )
        self.kernel(
            mA,
            mB,
            mD,
            mInfo,
            self.sA_layout,
            self.sB_layout,
            self.tiled_copy_a,
            self.tiled_copy_b,
            self.tiled_mma,
            self.tiled_copy_a_s2r,
            self.tiled_copy_b_s2r,
            SharedStorage,
        ).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
        )

    @cute.jit
    def load(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        sA: cute.Tensor,
        sB: cute.Tensor,
        tiled_copy_a: cute.TiledCopy,
        tiled_copy_b: cute.TiledCopy,
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
        m_idx = tile_m_idx * self.tile_m + local_m
        n_idx = tile_n_idx * self.tile_n + local_n

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
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
        gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

        a_thr_copy = tiled_copy_a.get_slice(producer_thread_idx)
        tAgA = a_thr_copy.partition_S(gA)
        tAsA = a_thr_copy.partition_D(sA)
        b_thr_copy = tiled_copy_b.get_slice(producer_thread_idx)
        tBgB = b_thr_copy.partition_S(gB)
        tBsB = b_thr_copy.partition_D(sB)

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
        tApA.fill(m_idx < valid_m)
        tBpB.fill(n_idx < n)

        for k_tile_idx in cutlass.range(k_tile_count):
            a_pipeline.producer_acquire(a_producer_state)
            cute.copy(
                tiled_copy_a,
                tAgA[None, None, None, k_tile_idx],
                tAsA[None, None, None, a_producer_state.index],
                pred=tApA,
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
            b_pipeline.producer_commit(b_producer_state)
            b_producer_state.advance()

        return a_producer_state, b_producer_state

    @cute.jit
    def compute(
        self,
        sA: cute.Tensor,
        sB: cute.Tensor,
        tiled_mma: cute.TiledMma,
        tiled_copy_a_s2r: cute.TiledCopy,
        tiled_copy_b_s2r: cute.TiledCopy,
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

        acc_shape = thr_mma.partition_shape_C((self.tile_m, self.tile_n))
        accumulators = cute.make_rmem_tensor(acc_shape, self.dtype_acc)
        accumulators.fill(0.0)

        for _ in cutlass.range(k_tile_count):
            a_pipeline.consumer_wait(a_consumer_state)
            cute.copy(
                tiled_copy_a_s2r,
                tAsA[None, None, 0, a_consumer_state.index],
                tArA[None, None, 0],
            )
            a_pipeline.consumer_release(a_consumer_state)
            a_consumer_state.advance()

            b_pipeline.consumer_wait(b_consumer_state)
            cute.copy(
                tiled_copy_b_s2r,
                tBsB[None, None, 0, b_consumer_state.index],
                tBrB[None, None, 0],
            )
            b_pipeline.consumer_release(b_consumer_state)
            b_consumer_state.advance()

            cute.gemm(
                tiled_mma,
                accumulators,
                tCrA[None, None, 0],
                tCrB[None, None, 0],
                accumulators,
            )
        return accumulators, a_consumer_state, b_consumer_state

    @cute.jit
    def epilogue(
        self,
        accumulators: cute.Tensor,
        mD: cute.Tensor,
        valid_m: cutlass.Int32,
        n: cutlass.Int32,
        tile_m_idx: cutlass.Int32,
        tile_n_idx: cutlass.Int32,
        group_idx: cutlass.Int32,
    ) -> None:
        lane_idx = cute.arch.lane_idx()
        tile_m_offset = tile_m_idx * self.tile_m
        tile_n_offset = tile_n_idx * self.tile_n

        rD = cute.make_fragment_like(accumulators, self.dtype_d)
        rD.store(accumulators.load().to(self.dtype_d))
        rD_mn = make_acc_tensor_mn_view(rD)
        assert cute.size(rD_mn, mode=[0]) == 2
        assert cute.size(rD_mn, mode=[1]) == 4

        lane_m = lane_idx // 4
        lane_n = (lane_idx % 4) * 2
        for row in cutlass.range_constexpr(cute.size(rD_mn, mode=[0])):
            m_idx = tile_m_offset + lane_m + row * 8
            for col in cutlass.range_constexpr(cute.size(rD_mn, mode=[1])):
                n_idx = tile_n_offset + lane_n + col % 2 + (col // 2) * 8
                if m_idx < valid_m and n_idx < n:
                    mD[group_idx, m_idx, n_idx] = rD_mn[row, col]

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        mInfo: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_copy_a: cute.TiledCopy,
        tiled_copy_b: cute.TiledCopy,
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
                    mB,
                    sA,
                    sB,
                    tiled_copy_a,
                    tiled_copy_b,
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
                accumulators, a_consumer_state, b_consumer_state = self.compute(
                    sA,
                    sB,
                    tiled_mma,
                    tiled_copy_a_s2r,
                    tiled_copy_b_s2r,
                    k_tile_count,
                    a_pipeline,
                    b_pipeline,
                    a_consumer_state,
                    b_consumer_state,
                )
                self.epilogue(accumulators, mD, valid_m, n, tile_m_idx, tile_n_idx, group_idx)
            work_tile = scheduler.advance_to_next_work()

        if is_producer:
            a_pipeline.producer_tail(a_producer_state)
            b_pipeline.producer_tail(b_producer_state)
