import cutlass
import cutlass.cute as cute

from .utils import make_acc_tensor_mn_view


@cute.jit
def online_softmax(
    acc_S: cute.Tensor,
    acc_O: cute.Tensor,
    row_max: cute.Tensor,
    row_sum: cute.Tensor,
    softmax_scale_log2: cutlass.Float32,
) -> None:
    acc_S_mn = make_acc_tensor_mn_view(acc_S)
    acc_O_mn = make_acc_tensor_mn_view(acc_O)

    for row in cutlass.range_constexpr(cute.size(row_max)):
        scores = acc_S_mn[row, None].load()
        tile_max = scores.reduce(cute.ReductionOp.MAX, -cutlass.Float32.inf, 0)
        new_max = cute.arch.warp_reduction_max(
            cute.arch.fmax(row_max[row], tile_max),
            threads_in_group=4,
        )
        safe_max = 0.0 if new_max == -cutlass.Float32.inf else new_max
        correction = cute.math.exp2(
            (row_max[row] - safe_max) * softmax_scale_log2,
            fastmath=True,
        )
        probabilities = cute.math.exp2(
            scores * softmax_scale_log2 - safe_max * softmax_scale_log2,
            fastmath=True,
        )
        tile_sum = probabilities.reduce(
            cute.ReductionOp.ADD,
            cutlass.Float32.zero,
            0,
        )

        row_max[row] = new_max
        row_sum[row] = row_sum[row] * correction + tile_sum
        acc_S_mn[row, None].store(probabilities)
        acc_O_mn[row, None].store(acc_O_mn[row, None].load() * correction)


@cute.jit
def normalize_output(acc_O: cute.Tensor, row_sum: cute.Tensor) -> None:
    acc_O_mn = make_acc_tensor_mn_view(acc_O)

    for row in cutlass.range_constexpr(cute.size(row_sum)):
        total = cute.arch.warp_reduction_sum(row_sum[row], threads_in_group=4)
        invalid_total = total == 0.0 or total != total
        inverse_total = 1.0 if invalid_total else cute.arch.rcp_approx(total)
        acc_O_mn[row, None].store(acc_O_mn[row, None].load() * inverse_total)
