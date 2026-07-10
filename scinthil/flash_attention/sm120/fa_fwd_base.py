class FlashAttentionForwardBase:
    def __init__(
        self,
        dtype,
        headdim_qk: int,
        headdim_v: int,
        tile_m: int,
        tile_n: int,
        is_causal: bool = False,
    ) -> None:
        self.dtype = dtype
        self.acc_dtype = dtype
        self.headdim_qk = headdim_qk
        self.headdim_v = headdim_v
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.is_causal = is_causal

    def get_smem_layout_atom(self):
        raise NotImplementedError

    def get_smem_layout(self):
        raise NotImplementedError

    def get_qk_pv_mma_atom(self):
        raise NotImplementedError

    def get_qk_pv_tiled_mma(self):
        raise NotImplementedError
