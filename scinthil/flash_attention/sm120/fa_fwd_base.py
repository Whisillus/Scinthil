import cutlass


class FlashAttentionForwardBase:
    def __init__(
        self,
        dtype,
        headdim_qk: int,
        headdim_v: int,
        tile_m: int,
        tile_n: int,
        stage_k: int,
        stage_v: int,
        is_causal: bool = False,
    ) -> None:
        if dtype not in (cutlass.Float16, cutlass.BFloat16):
            raise TypeError("FlashAttentionForwardBase supports only Float16 and BFloat16")
        if headdim_qk % 64 != 0 or headdim_v % 64 != 0:
            raise ValueError("QK and V head dimensions must be multiples of 64")
        if stage_k <= 0 or stage_v <= 0:
            raise ValueError("K and V stage counts must be positive")

        self.dtype = dtype
        self.dtype_byte = cutlass.const_expr(self.dtype.width // 8)
        self.acc_dtype = dtype
        self.headdim_qk = headdim_qk
        self.headdim_v = headdim_v
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.stage_k = stage_k
        self.stage_v = stage_v
        self.is_causal = is_causal

    def can_implement(self) -> bool:
        raise NotImplementedError

    def get_smem_layout_atom(self) -> None:
        raise NotImplementedError

    def get_smem_layout(self) -> None:
        raise NotImplementedError

    def get_qk_pv_mma_atom(self):
        raise NotImplementedError

    def get_qk_pv_tiled_mma(self):
        raise NotImplementedError
