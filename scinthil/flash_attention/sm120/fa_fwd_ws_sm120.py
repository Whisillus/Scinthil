import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync

from .block import BlockInfo
from .fa_fwd_base import FlashAttentionForwardBase
from .fa_utils import get_log2, get_predicate_load_v_seqlen, make_acc_tensor_mn_view, transform_frg_P
from .mask import FlashAttentionMaskSM120
from .softmax import FlashAttentionSoftmaxSM120
from .tile_scheduler import SingleTileScheduler


class FlashAttentionForwardM16N8K16SM120(FlashAttentionForwardBase):
    """Warp-specialized FlashAttention forward kernel configuration."""

    def get_warpspecialize_config(self) -> None:
        self.mma_inst_shape_mnk = (16, 8, 16)
        self.mma_inst_m, self.mma_inst_n, self.mma_inst_k = self.mma_inst_shape_mnk
        self.mma_warp_tile_shape_mnk = (self.mma_inst_m, 2 * self.mma_inst_n, self.mma_inst_k)

        assert self.tile_m % self.mma_warp_tile_shape_mnk[0] == 0, "tile_m must be divisible by warp MMA tile M"
        assert self.tile_n % self.mma_warp_tile_shape_mnk[1] == 0, "tile_n must be divisible by warp MMA tile N"

        self.num_producer = 1
        self.num_consumer = self.tile_m // self.mma_warp_tile_shape_mnk[0]
        self.threads_per_cta = (self.num_producer + self.num_consumer) * cute.arch.WARP_SIZE

        assert self.num_consumer > 0, "MMA configuration requires at least one consumer warp"
        assert self.threads_per_cta <= 1024, "MMA configuration exceeds the maximum CTA thread count"

    def can_implement(self) -> bool:
        if self.dtype not in (cutlass.Float16, cutlass.BFloat16):
            return False
        if self.acc_dtype != cutlass.Float32:
            return False
        if self.head_q <= 0 or self.head_kv <= 0:
            return False
        if self.head_q % self.head_kv != 0:
            return False
        if self.headdim_qk % 64 != 0 or self.headdim_v % 64 != 0:
            return False
        if self.stage_k <= 0 or self.stage_v <= 0:
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
        self.mma_atom_qk = cute.make_mma_atom(
            cute.nvgpu.warp.MmaF16BF16Op(
                self.dtype,
                self.acc_dtype,
                self.mma_inst_shape_mnk,
            )
        )
        self.mma_atom_pv = cute.make_mma_atom(
            cute.nvgpu.warp.MmaF16BF16Op(
                self.dtype,
                self.acc_dtype,
                self.mma_inst_shape_mnk,
            )
        )

    def get_qk_pv_tiled_mma(self) -> None:
        atom_shape_mnk = (self.num_consumer, 1, 1)
        atom_stride_mnk = (1, self.num_consumer, self.num_consumer)
        atom_layout_mnk = cute.make_layout(
            atom_shape_mnk,
            stride=atom_stride_mnk,
        )
        # Extend the native N=8 atom to N=16 so each warp can load the two
        # adjacent B fragments together with ldmatrix.x4 and reuse A for both MMAs.
        permutation_mnk = (
            self.num_consumer * self.mma_warp_tile_shape_mnk[0],
            self.mma_warp_tile_shape_mnk[1],
            self.mma_warp_tile_shape_mnk[2],
        )
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
        n_block_min: cutlass.Int32,
        n_block_max: cutlass.Int32,
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

        if n_block_min < n_block_max:
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

            for tile_n_idx in cutlass.range(n_block_max - 1, n_block_min - 1, -1):
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
    def compute_step(
        self,
        qk_thr_mma: cute.ThrMma,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_copy_q_s2r: cute.TiledCopy,
        tiled_copy_k_s2r: cute.TiledCopy,
        tiled_copy_v_s2r: cute.TiledCopy,
        tQsQ: cute.Tensor,
        tQrQ: cute.Tensor,
        tKsK: cute.Tensor,
        tKrK: cute.Tensor,
        tVsVt: cute.Tensor,
        tVrV: cute.Tensor,
        tCrQ: cute.Tensor,
        tCrK: cute.Tensor,
        tCrV: cute.Tensor,
        acc_o: cute.Tensor,
        softmax: FlashAttentionSoftmaxSM120,
        score_mask: FlashAttentionMaskSM120,
        tile_m_idx: cutlass.Int32,
        tile_n_idx: cutlass.Int32,
        seqlen_q: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        k_pipeline: pipeline.PipelineCpAsync,
        v_pipeline: pipeline.PipelineCpAsync,
        k_consumer_state: pipeline.PipelineState,
        v_consumer_state: pipeline.PipelineState,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState]:
        k_pipeline.consumer_wait(k_consumer_state)

        acc_shape_s = qk_thr_mma.partition_shape_C((self.tile_m, self.tile_n))
        acc_s = cute.make_rmem_tensor(acc_shape_s, self.acc_dtype)
        acc_s.fill(0.0)

        for qk_block in cutlass.range_constexpr(cute.size(tKsK.shape[2])):
            cute.copy(
                tiled_copy_q_s2r,
                tQsQ[None, None, qk_block],
                tQrQ[None, None, 0],
            )
            cute.copy(
                tiled_copy_k_s2r,
                tKsK[None, None, qk_block, k_consumer_state.index],
                tKrK[None, None, 0],
            )
            cute.gemm(
                tiled_mma_qk,
                acc_s,
                tCrQ[None, None, 0],
                tCrK[None, None, 0],
                acc_s,
            )

        k_pipeline.consumer_release(k_consumer_state)
        k_consumer_state.advance()

        score_mask.apply(
            acc_s,
            tile_m_idx,
            tile_n_idx,
            seqlen_q,
            seqlen_k,
        )
        row_scale = softmax.online_softmax(
            acc_s,
            is_first=is_first_n_block,
            check_inf=check_inf,
        )
        if cutlass.const_expr(not is_first_n_block):
            softmax.rescale_O(acc_o, row_scale)

        rP = cute.make_fragment_like(acc_s, self.dtype)
        rP.store(acc_s.load().to(self.dtype))
        tCrP = transform_frg_P(rP)

        assert cute.size(tCrP.shape[2]) == cute.size(tVsVt.shape[2])

        v_pipeline.consumer_wait(v_consumer_state)
        for pv_block in cutlass.range_constexpr(cute.size(tCrP.shape[2])):
            cute.copy(
                tiled_copy_v_s2r,
                tVsVt[None, None, pv_block, v_consumer_state.index],
                tVrV[None, None, 0],
            )
            cute.gemm(
                tiled_mma_pv,
                acc_o,
                tCrP[None, None, pv_block],
                tCrV[None, None, 0],
                acc_o,
            )
        v_pipeline.consumer_release(v_consumer_state)
        v_consumer_state.advance()

        return k_consumer_state, v_consumer_state

    @cute.jit
    def epilogue(
        self,
        acc_o: cute.Tensor,
        lse: cute.Tensor,
        tiled_mma_pv: cute.TiledMma,
        gO: cute.Tensor,
        gLSE: cute.Tensor,
        tile_m_idx: cutlass.Int32,
        seqlen_q: cutlass.Int32,
    ) -> None:
        consumer_thread_idx = cute.arch.thread_idx()[0]
        pv_thr_mma = tiled_mma_pv.get_slice(consumer_thread_idx)

        rO = cute.make_fragment_like(acc_o, self.dtype)
        rO.store(acc_o.load().to(self.dtype))

        cO = cute.make_identity_tensor((self.tile_m, self.headdim_v))
        tOgO_mn = make_acc_tensor_mn_view(pv_thr_mma.partition_C(gO))
        tOcO_mn = make_acc_tensor_mn_view(pv_thr_mma.partition_C(cO))
        tOrO_mn = make_acc_tensor_mn_view(rO)

        assert cute.size(tOrO_mn, mode=[0]) == cute.size(lse)

        for row in cutlass.range_constexpr(cute.size(tOrO_mn, mode=[0])):
            row_idx = tOcO_mn[row, 0][0]
            if cute.elem_less(tile_m_idx * self.tile_m + row_idx, seqlen_q):
                cute.autovec_copy(tOrO_mn[row, None], tOgO_mn[row, None])

        if tOcO_mn[0, 0][1] == 0:
            for row in cutlass.range_constexpr(cute.size(lse)):
                row_idx = tOcO_mn[row, 0][0]
                if cute.elem_less(tile_m_idx * self.tile_m + row_idx, seqlen_q):
                    gLSE[row_idx] = lse[row]

    @cute.jit
    def compute(
        self,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_copy_q_s2r: cute.TiledCopy,
        tiled_copy_k_s2r: cute.TiledCopy,
        tiled_copy_v_s2r: cute.TiledCopy,
        tile_m_idx: cutlass.Int32,
        seqlen_q: cutlass.Int32,
        seqlen_k: cutlass.Int32,
        softmax_scale_log2: cutlass.Float32,
        n_block_min: cutlass.Int32,
        n_block_max: cutlass.Int32,
        q_pipeline: pipeline.PipelineCpAsync,
        k_pipeline: pipeline.PipelineCpAsync,
        v_pipeline: pipeline.PipelineCpAsync,
    ) -> tuple[cute.Tensor, cute.Tensor]:
        consumer_thread_idx = cute.arch.thread_idx()[0]

        qk_thr_mma = tiled_mma_qk.get_slice(consumer_thread_idx)
        pv_thr_mma = tiled_mma_pv.get_slice(consumer_thread_idx)
        sQ_qk_block = cute.local_tile(sQ, (self.tile_m, self.mma_inst_k), (0, 0))
        sK_qk_block = cute.local_tile(sK[None, None, 0], (self.tile_n, self.mma_inst_k), (0, 0))
        tCsQ = qk_thr_mma.partition_A(sQ_qk_block)
        tCsK = qk_thr_mma.partition_B(sK_qk_block)
        tCrQ = qk_thr_mma.make_fragment_A(tCsQ)
        tCrK = qk_thr_mma.make_fragment_B(tCsK)

        q_thr_copy = tiled_copy_q_s2r.get_slice(consumer_thread_idx)
        tQsQ = q_thr_copy.partition_S(sQ)
        tQrQ = q_thr_copy.retile(tCrQ)

        k_thr_copy = tiled_copy_k_s2r.get_slice(consumer_thread_idx)
        tKsK = k_thr_copy.partition_S(sK)
        tKrK = k_thr_copy.retile(tCrK)

        assert cute.size(tQsQ.shape[2]) == cute.size(tKsK.shape[2])

        sVt = cute.composition(
            sV,
            cute.make_ordered_layout(
                (sV.shape[1], sV.shape[0], sV.shape[2]),
                order=(1, 0, 2),
            ),
        )
        sV_pv_block = cute.local_tile(sVt[None, None, 0], (self.headdim_v, self.mma_inst_k), (0, 0))
        tCsV = pv_thr_mma.partition_B(sV_pv_block)
        tCrV = pv_thr_mma.make_fragment_B(tCsV)

        v_thr_copy = tiled_copy_v_s2r.get_slice(consumer_thread_idx)
        tVsVt = v_thr_copy.partition_S(sVt)
        tVrV = v_thr_copy.retile(tCrV)

        acc_shape_o = pv_thr_mma.partition_shape_C((self.tile_m, self.headdim_v))
        acc_o = cute.make_rmem_tensor(acc_shape_o, self.acc_dtype)
        acc_o.fill(0.0)

        num_rows = acc_o.shape[0][0] * acc_o.shape[1]
        softmax = FlashAttentionSoftmaxSM120.create(
            softmax_scale_log2,
            num_rows,
            self.is_fastmath,
        )

        score_mask = FlashAttentionMaskSM120.create(qk_thr_mma, self.tile_m, self.tile_n, self.is_causal)

        q_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stage_q)
        k_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stage_k)
        v_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.stage_v)

        if n_block_min < n_block_max:
            q_pipeline.consumer_wait(q_consumer_state)
            tile_n_idx = n_block_max - 1
            k_consumer_state, v_consumer_state = self.compute_step(
                qk_thr_mma,
                tiled_mma_qk,
                tiled_mma_pv,
                tiled_copy_q_s2r,
                tiled_copy_k_s2r,
                tiled_copy_v_s2r,
                tQsQ,
                tQrQ,
                tKsK,
                tKrK,
                tVsVt,
                tVrV,
                tCrQ,
                tCrK,
                tCrV,
                acc_o,
                softmax,
                score_mask,
                tile_m_idx,
                tile_n_idx,
                seqlen_q,
                seqlen_k,
                k_pipeline,
                v_pipeline,
                k_consumer_state,
                v_consumer_state,
                is_first_n_block=True,
            )

            for tile_n_idx in cutlass.range(n_block_max - 2, n_block_min - 1, -1):
                k_consumer_state, v_consumer_state = self.compute_step(
                    qk_thr_mma,
                    tiled_mma_qk,
                    tiled_mma_pv,
                    tiled_copy_q_s2r,
                    tiled_copy_k_s2r,
                    tiled_copy_v_s2r,
                    tQsQ,
                    tQrQ,
                    tKsK,
                    tKrK,
                    tVsVt,
                    tVrV,
                    tCrQ,
                    tCrK,
                    tCrV,
                    acc_o,
                    softmax,
                    score_mask,
                    tile_m_idx,
                    tile_n_idx,
                    seqlen_q,
                    seqlen_k,
                    k_pipeline,
                    v_pipeline,
                    k_consumer_state,
                    v_consumer_state,
                    is_first_n_block=False,
                )

            q_pipeline.consumer_release(q_consumer_state)
            q_consumer_state.advance()

        final_scale = softmax.compute_final_scale()
        softmax.rescale_O(acc_o, final_scale)
        return acc_o, softmax.row_sum

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        softmax_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ) -> None:
        self.get_warpspecialize_config()
        self.get_smem_layout()
        self.get_qk_pv_mma_atom()
        self.get_qk_pv_tiled_mma()
        self.get_qkv_load()
        self.get_qkv_s2r()

        softmax_scale_log2 = softmax_scale * get_log2()

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
            softmax_scale_log2,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.tiled_copy_q,
            self.tiled_copy_k,
            self.tiled_copy_v,
            self.tiled_mma_qk,
            self.tiled_mma_pv,
            self.tiled_copy_q_s2r,
            self.tiled_copy_k_s2r,
            self.tiled_copy_v_s2r,
            SharedStorage,
        ).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        tiled_copy_q: cute.TiledCopy,
        tiled_copy_k: cute.TiledCopy,
        tiled_copy_v: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_copy_q_s2r: cute.TiledCopy,
        tiled_copy_k_s2r: cute.TiledCopy,
        tiled_copy_v_s2r: cute.TiledCopy,
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

        block_info = BlockInfo(self.tile_m, self.tile_n, self.is_causal)
        n_block_min, n_block_max = block_info.get_n_block_min_max(mQ.shape[1], mK.shape[1], tile_m_idx)
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
                n_block_min,
                n_block_max,
                q_pipeline,
                k_pipeline,
                v_pipeline,
            )
        if is_consumer:
            acc_o, lse = self.compute(
                sQ,
                sK,
                sV,
                tiled_mma_qk,
                tiled_mma_pv,
                tiled_copy_q_s2r,
                tiled_copy_k_s2r,
                tiled_copy_v_s2r,
                tile_m_idx,
                mQ.shape[1],
                mK.shape[1],
                softmax_scale_log2,
                n_block_min,
                n_block_max,
                q_pipeline,
                k_pipeline,
                v_pipeline,
            )
            self.epilogue(
                acc_o,
                lse,
                tiled_mma_pv,
                gO,
                gLSE,
                tile_m_idx,
                mQ.shape[1],
            )

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
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    )
