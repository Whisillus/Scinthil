import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync

from .group_gemm_base import GroupedGEMMBase
from .scheduler import PersistentTileScheduler


class MGroupedMaskedGEMMSM120(GroupedGEMMBase):
    def __init__(
        self,
        *,
        dtype,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        use_block_swizzle: bool,
        block_swizzle_factor: int = 8,
    ) -> None:
        super().__init__(
            dtype=dtype,
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

    def get_ab_load_atom(self) -> None:
        self.num_bits_per_copy = 128
        self.load_a_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=self.num_bits_per_copy,
        )
        self.load_b_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=self.num_bits_per_copy,
        )

    def get_ab_load(self) -> None:
        self.get_ab_load_atom()

        copy_elems = self.num_bits_per_copy // self.dtype.width
        self.a_vectors_per_row = self.sA_layout_atom.outer.shape[1] // copy_elems
        self.b_vectors_per_row = self.sB_layout_atom.outer.shape[1] // copy_elems
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
        value_layout = cute.make_layout(
            (1, copy_elems),
            stride=(copy_elems, 1),
        )
        self.tiled_copy_a = cute.make_tiled_copy_tv(self.load_a_atom, a_thread_layout, value_layout)
        self.tiled_copy_b = cute.make_tiled_copy_tv(self.load_b_atom, b_thread_layout, value_layout)

    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mD: cute.Tensor,
        mInfo: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        self.get_warpspecialize_config()
        self.get_smem_layout()
        self.get_ab_load()

        @cute.struct
        class SharedStorage:
            sA: cute.struct.Align[
                cute.struct.MemRange[self.dtype, self.sA_size],
                128,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.dtype, self.sB_size],
                128,
            ]
            ab_mbar: cute.struct.MemRange[cutlass.Int64, self.stages * 2]

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
        ab_pipeline: pipeline.PipelineCpAsync,
        producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
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
            ab_pipeline.producer_acquire(producer_state)
            cute.copy(
                tiled_copy_a,
                tAgA[None, None, None, k_tile_idx],
                tAsA[None, None, None, producer_state.index],
                pred=tApA,
            )
            cute.copy(
                tiled_copy_b,
                tBgB[None, None, None, k_tile_idx],
                tBsB[None, None, None, producer_state.index],
                pred=tBpB,
            )
            ab_pipeline.producer_commit(producer_state)
            producer_state.advance()

        return producer_state

    @cute.jit
    def compute(
        self,
        k_tile_count: cutlass.Int32,
        ab_pipeline: pipeline.PipelineCpAsync,
        consumer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        for _ in cutlass.range(k_tile_count):
            ab_pipeline.consumer_wait(consumer_state)
            ab_pipeline.consumer_release(consumer_state)
            consumer_state.advance()
        return consumer_state

    @cute.jit
    def epilogue(
        self,
        mD: cute.Tensor,
        valid_m: cutlass.Int32,
        n: cutlass.Int32,
        tile_m_idx: cutlass.Int32,
        tile_n_idx: cutlass.Int32,
        group_idx: cutlass.Int32,
    ) -> None:
        thread_idx = cute.arch.thread_idx()[0]
        tile_m_offset = tile_m_idx * self.tile_m
        tile_n_offset = tile_n_idx * self.tile_n

        for linear_idx in cutlass.range(
            thread_idx,
            self.tile_m * self.tile_n,
            self.consumer_threads,
        ):
            local_m = linear_idx // self.tile_n
            local_n = linear_idx % self.tile_n
            m_idx = tile_m_offset + local_m
            n_idx = tile_n_offset + local_n
            if m_idx < valid_m and n_idx < n:
                mD[group_idx, m_idx, n_idx] = self.dtype(0)

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
        ab_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.ab_mbar.data_ptr(),
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
        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            self.stages,
        )
        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.stages,
        )
        work_tile = scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            tile_m_idx, tile_n_idx, group_idx = work_tile.tile_idx
            valid_m = mInfo[group_idx]
            if is_producer:
                producer_state = self.load(
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
                    ab_pipeline,
                    producer_state,
                )
            if is_consumer:
                consumer_state = self.compute(
                    k_tile_count,
                    ab_pipeline,
                    consumer_state,
                )
                self.epilogue(mD, valid_m, n, tile_m_idx, tile_n_idx, group_idx)
            work_tile = scheduler.advance_to_next_work()

        if is_producer:
            ab_pipeline.producer_tail(producer_state)
