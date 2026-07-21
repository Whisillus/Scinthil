import cutlass
import cutlass.cute as cute
import cutlass.utils as utils


class WorkTileInfo(utils.WorkTileInfo):
    """Work tile coordinate in (tile_m, tile_n, group) order."""

    def __new_from_mlir_values__(self, values):
        assert len(values) == 4
        tile_idx = cutlass.new_from_mlir_values(self.tile_idx, values[:-1])
        is_valid_tile = cutlass.new_from_mlir_values(self.is_valid_tile, [values[-1]])
        return WorkTileInfo(tile_idx, is_valid_tile)


class PersistentTileScheduler:
    """Persistent masked-group scheduler with M-oriented L2 block swizzle."""

    @staticmethod
    def get_grid_shape(
        num_tile_m: cutlass.Int32,
        num_tile_n: cutlass.Int32,
        num_groups: cutlass.Int32,
        max_persistent_ctas: cutlass.Int32,
    ) -> tuple[cutlass.Int32, int, int]:
        total_tiles = num_tile_m * num_tile_n * num_groups
        return min(total_tiles, max_persistent_ctas), 1, 1

    def __init__(
        self,
        tile_m: int,
        use_block_swizzle: bool,
        num_1d_blocks_per_group: int,
        num_tile_m: cutlass.Int32,
        num_tile_n: cutlass.Int32,
        num_groups: cutlass.Int32,
        mInfo: cute.Tensor,
        current_linear_idx: cutlass.Int32,
        current_group_idx: cutlass.Int32,
        group_tile_start: cutlass.Int32,
        group_tile_end: cutlass.Int32,
        num_persistent_ctas: cutlass.Int32,
        *,
        loc=None,
        ip=None,
    ) -> None:
        self.tile_m = tile_m
        self.use_block_swizzle = use_block_swizzle
        self.num_1d_blocks_per_group = num_1d_blocks_per_group
        self.num_tile_m = num_tile_m
        self.num_tile_n = num_tile_n
        self.num_groups = num_groups
        self.mInfo = mInfo
        self.current_linear_idx = current_linear_idx
        self.current_group_idx = current_group_idx
        self.group_tile_start = group_tile_start
        self.group_tile_end = group_tile_end
        self.num_persistent_ctas = num_persistent_ctas
        self.loc = loc
        self.ip = ip

    @staticmethod
    @cute.jit
    def create(
        tile_m: int,
        use_block_swizzle: cutlass.Constexpr,
        num_1d_blocks_per_group: int,
        num_tile_m: cutlass.Int32,
        num_tile_n: cutlass.Int32,
        num_groups: cutlass.Int32,
        mInfo: cute.Tensor,
        block_idx_x: cutlass.Int32,
        grid_dim_x: cutlass.Int32,
        *,
        loc=None,
        ip=None,
    ) -> "PersistentTileScheduler":
        initial_group_tile_end = cute.ceil_div(mInfo[0], tile_m) * num_tile_n
        return PersistentTileScheduler(
            tile_m,
            use_block_swizzle,
            num_1d_blocks_per_group,
            num_tile_m,
            num_tile_n,
            num_groups,
            mInfo,
            block_idx_x,
            cutlass.Int32(0),
            cutlass.Int32(0),
            initial_group_tile_end,
            grid_dim_x,
            loc=loc,
            ip=ip,
        )

    @cute.jit
    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        while self.current_linear_idx >= self.group_tile_end and self.current_group_idx < self.num_groups:
            self.current_group_idx += 1
            self.group_tile_start = self.group_tile_end
            if self.current_group_idx < self.num_groups:
                self.group_tile_end += cute.ceil_div(self.mInfo[self.current_group_idx], self.tile_m) * self.num_tile_n

        tile_m_idx = cutlass.Int32(0)
        tile_n_idx = cutlass.Int32(0)
        group_idx = cutlass.Int32(0)
        is_valid_tile = self.current_group_idx < self.num_groups
        if is_valid_tile:
            current_num_tile_m = cute.ceil_div(self.mInfo[self.current_group_idx], self.tile_m)
            local_linear_idx = self.current_linear_idx - self.group_tile_start
            if cutlass.const_expr(self.use_block_swizzle):
                num_blocks_per_group = self.num_tile_n * self.num_1d_blocks_per_group
                block_group_idx = local_linear_idx // num_blocks_per_group
                first_m_block_idx = block_group_idx * self.num_1d_blocks_per_group
                in_group_idx = local_linear_idx % num_blocks_per_group
                num_blocks_in_group = cutlass.min(
                    self.num_1d_blocks_per_group,
                    current_num_tile_m - first_m_block_idx,
                )
                tile_m_idx = first_m_block_idx + in_group_idx % num_blocks_in_group
                tile_n_idx = in_group_idx // num_blocks_in_group
            else:
                tile_m_idx = local_linear_idx % current_num_tile_m
                tile_n_idx = local_linear_idx // current_num_tile_m
            group_idx = self.current_group_idx
        return WorkTileInfo((tile_m_idx, tile_n_idx, group_idx), is_valid_tile)

    @cute.jit
    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self.get_current_work(loc=loc, ip=ip)

    @cute.jit
    def advance_to_next_work(self, *, advance_count=1, loc=None, ip=None) -> WorkTileInfo:
        self.current_linear_idx += advance_count * self.num_persistent_ctas
        return self.get_current_work(loc=loc, ip=ip)

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.current_linear_idx)
        values.extend(cutlass.extract_mlir_values(self.current_group_idx))
        values.extend(cutlass.extract_mlir_values(self.group_tile_start))
        values.extend(cutlass.extract_mlir_values(self.group_tile_end))
        return values

    def __new_from_mlir_values__(self, values):
        assert len(values) == 4
        current_linear_idx = cutlass.new_from_mlir_values(self.current_linear_idx, values[:1])
        current_group_idx = cutlass.new_from_mlir_values(self.current_group_idx, values[1:2])
        group_tile_start = cutlass.new_from_mlir_values(self.group_tile_start, values[2:3])
        group_tile_end = cutlass.new_from_mlir_values(self.group_tile_end, values[3:])
        return PersistentTileScheduler(
            self.tile_m,
            self.use_block_swizzle,
            self.num_1d_blocks_per_group,
            self.num_tile_m,
            self.num_tile_n,
            self.num_groups,
            self.mInfo,
            current_linear_idx,
            current_group_idx,
            group_tile_start,
            group_tile_end,
            self.num_persistent_ctas,
            loc=self.loc,
            ip=self.ip,
        )
