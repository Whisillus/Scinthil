import cutlass
import cutlass.cute as cute


class FlashAttentionForwardBase:
    def __init__(
        self,
        dtype,
        head_q: int,
        head_kv: int,
        headdim_qk: int,
        headdim_v: int,
        tile_m: int,
        tile_n: int,
        stage_k: int = 1,
        stage_v: int = 1,
        is_causal: bool = False,
        num_producer: int = 1,
        num_consumer: int = 1,
        acc_dtype=cutlass.Float32,
    ) -> None:
        self.dtype = dtype
        self.dtype_byte = cutlass.const_expr(self.dtype.width // 8)
        self.acc_dtype = acc_dtype
        self.head_q = head_q
        self.head_kv = head_kv
        self.headdim_qk = headdim_qk
        self.headdim_v = headdim_v
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.stage_k = stage_k
        self.stage_v = stage_v
        self.is_causal = is_causal
        self.num_producer = num_producer
        self.num_consumer = num_consumer
        self.threads_per_cta = (self.num_producer + self.num_consumer) * cute.arch.WARP_SIZE

    def can_implement(self) -> bool:
        raise NotImplementedError

    def get_smem_layout_atom(self) -> None:
        raise NotImplementedError

    def get_smem_layout(self) -> None:
        raise NotImplementedError

    def get_qk_pv_mma_atom(self) -> None:
        raise NotImplementedError

    def get_qk_pv_tiled_mma(self) -> None:
        raise NotImplementedError
