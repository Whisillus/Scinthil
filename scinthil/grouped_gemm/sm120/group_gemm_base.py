from typing import Any

import cutlass


class GroupedGEMMBase:
    def __init__(
        self,
        *,
        dtype: type[Any],
        tile_m: int,
        tile_n: int,
        tile_k: int,
        use_block_swizzle: bool,
        block_swizzle_factor: int = 8,
        stages: int = 1,
        acc_dtype: type[Any] = cutlass.Float32,
        max_persistent_ctas: int = 170,
    ) -> None:
        self.dtype = dtype
        self.dtype_byte = cutlass.const_expr(self.dtype.width // 8)
        self.acc_dtype = acc_dtype
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.use_block_swizzle = use_block_swizzle
        self.block_swizzle_factor = block_swizzle_factor
        self.stages = stages
        self.max_persistent_ctas = max_persistent_ctas

    def can_implement(self) -> bool:
        raise NotImplementedError

    def get_warpspecialize_config(self) -> None:
        raise NotImplementedError

    def get_smem_layout_atom(self) -> None:
        raise NotImplementedError

    def get_smem_layout(self) -> None:
        raise NotImplementedError

    def get_ab_load_atom(self) -> None:
        raise NotImplementedError

    def get_ab_load(self) -> None:
        raise NotImplementedError

    def get_ab_s2r_atom(self) -> None:
        raise NotImplementedError

    def get_ab_s2r(self) -> None:
        raise NotImplementedError

    def get_mma_atom(self) -> None:
        raise NotImplementedError

    def get_tiled_mma(self) -> None:
        raise NotImplementedError
