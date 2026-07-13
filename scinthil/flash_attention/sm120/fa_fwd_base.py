import cutlass


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
        acc_dtype=cutlass.Float32,
    ) -> None:
        self.dtype = dtype
        self.dtype_byte = cutlass.const_expr(self.dtype.width // 8)
        self.acc_dtype = acc_dtype
        self.head_q = head_q
        self.head_kv = head_kv
        self.qhead_per_kvhead = self.head_q // self.head_kv if self.head_kv > 0 else 0
        self.headdim_qk = headdim_qk
        self.headdim_v = headdim_v
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.stage_q = 1
        self.stage_k = stage_k
        self.stage_v = stage_v
        self.is_causal = is_causal

    def can_implement(self) -> bool:
        raise NotImplementedError

    def get_smem_layout_atom(self) -> None:
        raise NotImplementedError

    def get_smem_layout(self) -> None:
        raise NotImplementedError

    def get_qkv_load_atom(self) -> None:
        raise NotImplementedError

    def get_qkv_load(self) -> None:
        raise NotImplementedError

    def get_qkv_s2r_atom(self) -> None:
        raise NotImplementedError

    def get_qkv_s2r(self) -> None:
        raise NotImplementedError

    def get_qk_pv_mma_atom(self) -> None:
        raise NotImplementedError

    def get_qk_pv_tiled_mma(self) -> None:
        raise NotImplementedError
