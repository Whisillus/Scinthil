import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync

from .fa_fwd_base import FlashAttentionForwardBase
from .fa_utils import get_predicate_load_v_seqlen
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
        if self.head_q <= 0 or self.head_kv <= 0:
            return False
        if self.head_q % self.head_kv != 0:
            return False
        if self.headdim_qk % 64 != 0 or self.headdim_v % 64 != 0:
            return False
        if self.tile_m % 16 != 0 or self.tile_n % 16 != 0:
            return False
        if self.stage_k <= 0 or self.stage_v <= 0:
            return False
        if self.num_producer <= 0 or self.num_consumer <= 0:
            return False
        if self.threads_per_cta > 1024:
            return False

        smem_usage_q = self.tile_m * self.headdim_qk * self.dtype_byte
        smem_usage_k = self.tile_n * self.headdim_qk * self.stage_k * self.dtype_byte
        smem_usage_v = self.tile_n * self.headdim_v * self.stage_v * self.dtype_byte
        smem_usage_barrier = 2 * (self.stage_q + self.stage_k + self.stage_v) * 8
        smem_usage = smem_usage_q + smem_usage_k + smem_usage_v + smem_usage_barrier

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
        self.sQ_size = cute.cosize(self.sQ_layout)
        self.sK_size = cute.cosize(self.sK_layout)
        self.sV_size = cute.cosize(self.sV_layout)

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

    def get_qkv_load_atom(self) -> None:
        self.num_bits_per_copy = 128
        self.load_q_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=self.num_bits_per_copy,
        )
        self.load_k_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=self.num_bits_per_copy,
        )
        self.load_v_atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=self.num_bits_per_copy,
        )

    def get_qkv_load(self) -> None:
        self.get_qkv_load_atom()

        copy_elems = self.num_bits_per_copy // self.dtype.width
        producer_threads = self.num_producer * cute.arch.WARP_SIZE
        value_layout = cute.make_layout((1, copy_elems))

        q_vectors_per_row = self.sQ_layout_atom.outer.shape[1] // copy_elems
        k_vectors_per_row = self.sK_layout_atom.outer.shape[1] // copy_elems
        v_vectors_per_row = self.sV_layout_atom.outer.shape[1] // copy_elems

        assert producer_threads % q_vectors_per_row == 0, "Producer threads must cover complete Q rows"
        assert producer_threads % k_vectors_per_row == 0, "Producer threads must cover complete K rows"
        assert producer_threads % v_vectors_per_row == 0, "Producer threads must cover complete V rows"

        q_rows_per_copy = producer_threads // q_vectors_per_row
        k_rows_per_copy = producer_threads // k_vectors_per_row
        v_rows_per_copy = producer_threads // v_vectors_per_row

        assert self.tile_m % q_rows_per_copy == 0, "Q copy rows must evenly divide tile_m"
        assert self.tile_n % k_rows_per_copy == 0, "K copy rows must evenly divide tile_n"
        assert self.tile_n % v_rows_per_copy == 0, "V copy rows must evenly divide tile_n"

        q_thread_layout = cute.make_layout((q_rows_per_copy, q_vectors_per_row), stride=(q_vectors_per_row, 1))
        self.tiled_copy_q = cute.make_tiled_copy_tv(
            self.load_q_atom,
            q_thread_layout,
            value_layout,
        )

        k_thread_layout = cute.make_layout((k_rows_per_copy, k_vectors_per_row), stride=(k_vectors_per_row, 1))
        self.tiled_copy_k = cute.make_tiled_copy_tv(
            self.load_k_atom,
            k_thread_layout,
            value_layout,
        )

        v_thread_layout = cute.make_layout((v_rows_per_copy, v_vectors_per_row), stride=(v_vectors_per_row, 1))
        self.tiled_copy_v = cute.make_tiled_copy_tv(
            self.load_v_atom,
            v_thread_layout,
            value_layout,
        )

    def get_qkv_s2r_atom(self) -> None:
        # CuTe MMA uses A(M,K) and B(N,K): Q(M,D) and K(N,D) already
        # match those views, while V(K,Dv) must transpose into B(Dv,K).
        self.s2r_q_atom = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.dtype,
        )
        self.s2r_k_atom = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.dtype,
        )
        self.s2r_v_atom = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self.dtype,
        )

    def get_qkv_s2r(self) -> None:
        self.get_qkv_s2r_atom()
        self.tiled_copy_q_s2r = cute.make_tiled_copy_A(self.s2r_q_atom, self.tiled_mma_qk)
        self.tiled_copy_k_s2r = cute.make_tiled_copy_B(self.s2r_k_atom, self.tiled_mma_qk)
        self.tiled_copy_v_s2r = cute.make_tiled_copy_B(self.s2r_v_atom, self.tiled_mma_pv)

    @cute.jit
    def load(
        self,
        gQ: cute.Tensor,
        gK: cute.Tensor,
        gV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tiled_copy_q: cute.TiledCopy,
        tiled_copy_k: cute.TiledCopy,
        tiled_copy_v: cute.TiledCopy,
        tile_m_idx: cutlass.Int32,
        seqlen_q: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        num_tile_n: cutlass.Int32,
        q_pipeline: pipeline.PipelineCpAsync,
        k_pipeline: pipeline.PipelineCpAsync,
        v_pipeline: pipeline.PipelineCpAsync,
    ) -> None:
        producer_thread_idx = cute.arch.thread_idx()[0] - self.num_consumer * cute.arch.WARP_SIZE

        q_thr_copy = tiled_copy_q.get_slice(producer_thread_idx)
        tQgQ = q_thr_copy.partition_S(gQ)
        tQsQ = q_thr_copy.partition_D(sQ)

        cQ = cute.make_identity_tensor((self.tile_m, self.headdim_qk))
        tQcQ = q_thr_copy.partition_S(cQ)

        k_thr_copy = tiled_copy_k.get_slice(producer_thread_idx)
        tKgK = k_thr_copy.partition_S(gK)
        tKsK = k_thr_copy.partition_D(sK)

        cK = cute.make_identity_tensor((self.tile_n, self.headdim_qk))
        tKcK = k_thr_copy.partition_S(cK)

        v_thr_copy = tiled_copy_v.get_slice(producer_thread_idx)
        tVgV = v_thr_copy.partition_S(gV)
        tVsV = v_thr_copy.partition_D(sV)

        cV = cute.make_identity_tensor((self.tile_n, self.headdim_v))
        tVcV = v_thr_copy.partition_S(cV)

        q_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stage_q)
        k_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stage_k)
        v_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.stage_v)

        q_pipeline.producer_acquire(q_producer_state)
        for q_row in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            q_row_coord = tQcQ[(0, 0), q_row, 0][0]
            if cute.elem_less(tile_m_idx * self.tile_m + q_row_coord, seqlen_q):
                cute.copy(
                    tiled_copy_q,
                    tQgQ[None, q_row, None],
                    tQsQ[None, q_row, None],
                )
        q_pipeline.producer_commit(q_producer_state)
        q_producer_state.advance()

        for tile_n_idx in cutlass.range(0, num_tile_n, 1):
            k_pipeline.producer_acquire(k_producer_state)
            for k_row in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                k_row_coord = tKcK[(0, 0), k_row, 0][0]
                if cute.elem_less(tile_n_idx * self.tile_n + k_row_coord, seqlen_k):
                    cute.copy(
                        tiled_copy_k,
                        tKgK[None, k_row, None, tile_n_idx],
                        tKsK[None, k_row, None, k_producer_state.index],
                    )
            k_pipeline.producer_commit(k_producer_state)
            k_producer_state.advance()

            # PV consumes the full V tile; zfill prevents stale or NaN SMEM values
            # from propagating through 0 * NaN at masked sequence positions.
            tVpV = get_predicate_load_v_seqlen(tVcV, tile_n_idx, seqlen_k, self.tile_n)

            v_pipeline.producer_acquire(v_producer_state)
            cute.copy(
                tiled_copy_v,
                tVgV[None, None, None, tile_n_idx],
                tVsV[None, None, None, v_producer_state.index],
                pred=tVpV,
            )
            v_pipeline.producer_commit(v_producer_state)
            v_producer_state.advance()

        q_pipeline.producer_tail(q_producer_state)
        k_pipeline.producer_tail(k_producer_state)
        v_pipeline.producer_tail(v_producer_state)

    @cute.jit
    def compute(
        self,
        num_tile_n: cutlass.Int32,
        q_pipeline: pipeline.PipelineCpAsync,
        k_pipeline: pipeline.PipelineCpAsync,
        v_pipeline: pipeline.PipelineCpAsync,
    ) -> None:
        q_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stage_q)
        k_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stage_k)
        v_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stage_v)

        q_pipeline.consumer_wait(q_consumer_state)
        q_pipeline.consumer_release(q_consumer_state)
        q_consumer_state.advance()

        for _ in cutlass.range(0, num_tile_n, 1):
            k_pipeline.consumer_wait(k_consumer_state)
            k_pipeline.consumer_release(k_consumer_state)
            k_consumer_state.advance()

            v_pipeline.consumer_wait(v_consumer_state)
            v_pipeline.consumer_release(v_consumer_state)
            v_consumer_state.advance()

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        softmax_scale: cutlass.Float32,
    ) -> None:
        self.get_smem_layout()
        self.get_qk_pv_tiled_mma()
        self.get_qkv_load()
        self.get_qkv_s2r()

        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[cute.struct.MemRange[self.dtype, self.sQ_size], 128]
            sK: cute.struct.Align[cute.struct.MemRange[self.dtype, self.sK_size], 128]
            sV: cute.struct.Align[cute.struct.MemRange[self.dtype, self.sV_size], 128]
            q_mbar: cute.struct.MemRange[cutlass.Int64, self.stage_q * 2]
            k_mbar: cute.struct.MemRange[cutlass.Int64, self.stage_k * 2]
            v_mbar: cute.struct.MemRange[cutlass.Int64, self.stage_v * 2]

        num_batch, seqlen_q, _, _ = mQ.shape
        num_tile_m = cute.ceil_div(seqlen_q, self.tile_m)
        grid = SingleTileScheduler.get_grid_shape(
            num_tile_m,
            self.head_q,
            num_batch,
        )
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            softmax_scale,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.tiled_copy_q,
            self.tiled_copy_k,
            self.tiled_copy_v,
            SharedStorage,
        ).launch(
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
        mLSE: cute.Tensor,
        softmax_scale: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        tiled_copy_q: cute.TiledCopy,
        tiled_copy_k: cute.TiledCopy,
        tiled_copy_v: cute.TiledCopy,
        SharedStorage: cutlass.Constexpr,
    ) -> None:
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        is_consumer = warp_idx < self.num_consumer
        is_producer = warp_idx >= self.num_consumer
        is_producer &= warp_idx < self.num_consumer + self.num_producer

        tile_scheduler = SingleTileScheduler.create()
        work_tile = tile_scheduler.initial_work_tile_info()
        tile_m_idx, head_q_idx, batch_idx, _ = work_tile.tile_idx

        head_kv_idx = head_q_idx // self.qhead_per_kvhead

        gQ = cute.local_tile(
            mQ[batch_idx, None, head_q_idx, None],
            (self.tile_m, self.headdim_qk),
            (tile_m_idx, 0),
        )
        gK = cute.local_tile(
            mK[batch_idx, None, head_kv_idx, None],
            (self.tile_n, self.headdim_qk),
            (None, 0),
        )
        gV = cute.local_tile(
            mV[batch_idx, None, head_kv_idx, None],
            (self.tile_n, self.headdim_v),
            (None, 0),
        )
        gO = cute.local_tile(
            mO[batch_idx, None, head_q_idx, None],
            (self.tile_m, self.headdim_v),
            (tile_m_idx, 0),
        )
        gLSE = cute.local_tile(
            mLSE[batch_idx, head_q_idx, None],
            (self.tile_m,),
            (tile_m_idx,),
        )

        assert cute.rank(gQ) == cute.rank(gO) == 2
        assert cute.rank(gK) == cute.rank(gV) == 3
        assert cute.rank(gLSE) == 1

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = storage.sQ.get_tensor(sQ_layout)
        sK = storage.sK.get_tensor(sK_layout)
        sV = storage.sV.get_tensor(sV_layout)

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_producer * cute.arch.WARP_SIZE)
        consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_consumer * cute.arch.WARP_SIZE)
        q_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.q_mbar.data_ptr(),
            num_stages=self.stage_q,
            producer_group=producer_group,
            consumer_group=consumer_group,
            defer_sync=True,
        )
        k_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.k_mbar.data_ptr(),
            num_stages=self.stage_k,
            producer_group=producer_group,
            consumer_group=consumer_group,
            defer_sync=True,
        )
        v_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.v_mbar.data_ptr(),
            num_stages=self.stage_v,
            producer_group=producer_group,
            consumer_group=consumer_group,
            defer_sync=True,
        )
        pipeline.pipeline_init_arrive()
        pipeline.pipeline_init_wait()

        num_tile_n = cute.size(gK.shape[2])
        if is_producer:
            self.load(
                gQ,
                gK,
                gV,
                sQ,
                sK,
                sV,
                tiled_copy_q,
                tiled_copy_k,
                tiled_copy_v,
                tile_m_idx,
                mQ.shape[1],
                mK.shape[1],
                num_tile_n,
                q_pipeline,
                k_pipeline,
                v_pipeline,
            )
        if is_consumer:
            self.compute(num_tile_n, q_pipeline, k_pipeline, v_pipeline)

        assert cute.rank(sQ) == 2
        assert cute.rank(sK) == cute.rank(sV) == 3


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
        (1, fa_fwd.tile_m, fa_fwd.head_q, fa_fwd.headdim_qk),
        **tensor_options,
    )
    mK = make_tensor(
        fa_fwd.dtype,
        (1, fa_fwd.tile_n, fa_fwd.head_kv, fa_fwd.headdim_qk),
        **tensor_options,
    )
    mV = make_tensor(
        fa_fwd.dtype,
        (1, fa_fwd.tile_n, fa_fwd.head_kv, fa_fwd.headdim_v),
        **tensor_options,
    )
    mO = make_tensor(
        fa_fwd.dtype,
        (1, fa_fwd.tile_m, fa_fwd.head_q, fa_fwd.headdim_v),
        **tensor_options,
    )
    mLSE = make_tensor(
        cutlass.Float32,
        (1, fa_fwd.head_q, fa_fwd.tile_m),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    compile_options = (cute.GPUArch("sm_120a"), cute.EnableTVMFFI)
    return cute.compile[compile_options](
        fa_fwd,
        mQ,
        mK,
        mV,
        mO,
        mLSE,
        cutlass.Float32(1.0),
    )
