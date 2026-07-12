import cutlass
import cutlass.cute as cute
import cutlass.utils as utils


class WorkTileInfo(utils.WorkTileInfo):
    """Work tile coordinate in (tile_m, head, batch, split) order."""

    def __new_from_mlir_values__(self, values):
        assert len(values) == 5
        tile_idx = cutlass.new_from_mlir_values(self.tile_idx, values[:-1])
        is_valid_tile = cutlass.new_from_mlir_values(self.is_valid_tile, [values[-1]])
        return WorkTileInfo(tile_idx, is_valid_tile)


class SingleTileScheduler:
    """Assign the current CTA's work tile exactly once."""

    @staticmethod
    def get_grid_shape(
        num_tile_m: cutlass.Int32,
        num_head: cutlass.Int32,  # num_head_q normally, num_head_kv for packed GQA
        num_batch: cutlass.Int32,
    ) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
        """Return the launch grid for single-tile scheduling.

        Args:
            num_tile_m: Number of query-sequence tiles along M.
            num_head: Number of Q heads normally or KV heads for packed GQA.
            num_batch: Number of batches.

        Returns:
            Grid shape `(num_tile_m, num_head, num_batch)`.
        """
        return num_tile_m, num_head, num_batch

    def __init__(self, tile_idx: cute.Coord, *, loc=None, ip=None) -> None:
        self.tile_idx = tile_idx
        self.is_first_tile = True
        self.loc = loc
        self.ip = ip

    @staticmethod
    def create(*, loc=None, ip=None) -> "SingleTileScheduler":
        # head_idx is a Q head normally and a KV head for packed GQA.
        tile_m_idx, head_idx, batch_idx = cute.arch.block_idx()
        tile_idx = (tile_m_idx, head_idx, batch_idx, cutlass.Int32(0))
        return SingleTileScheduler(tile_idx, loc=loc, ip=ip)

    def get_current_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        return WorkTileInfo(self.tile_idx, self.is_first_tile)

    def initial_work_tile_info(self, *, loc=None, ip=None) -> WorkTileInfo:
        return self.get_current_work(loc=loc, ip=ip)

    def prefetch_next_work(self, *, loc=None, ip=None) -> None:
        return None

    def advance_to_next_work(self, *, loc=None, ip=None) -> WorkTileInfo:
        self.is_first_tile = False
        return self.get_current_work(loc=loc, ip=ip)

    def producer_tail(self, *, loc=None, ip=None) -> None:
        return None

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.tile_idx)

    def __new_from_mlir_values__(self, values):
        tile_idx = cutlass.new_from_mlir_values(self.tile_idx, values)
        return SingleTileScheduler(tile_idx, loc=self.loc, ip=self.ip)
