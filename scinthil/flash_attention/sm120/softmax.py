from dataclasses import dataclass

import cutlass
import cutlass.cute as cute

from .fa_utils import make_acc_tensor_mn_view


@cute.jit
def fmax_reduce(values: cute.TensorSSA, init_val: cutlass.Float32) -> cutlass.Float32:
    num_values = cute.size(values.shape)
    assert num_values >= 4 and num_values % 4 == 0

    fragment = cute.make_rmem_tensor(values.shape, cutlass.Float32)
    fragment.store(values)

    local_max = [fragment[0], fragment[1], fragment[2], fragment[3]]
    for idx in cutlass.range_constexpr(4, num_values, 4):
        local_max[0] = cute.arch.fmax(local_max[0], fragment[idx])
        local_max[1] = cute.arch.fmax(local_max[1], fragment[idx + 1])
        local_max[2] = cute.arch.fmax(local_max[2], fragment[idx + 2])
        local_max[3] = cute.arch.fmax(local_max[3], fragment[idx + 3])

    local_max[0] = cute.arch.fmax(local_max[0], local_max[1])
    local_max[2] = cute.arch.fmax(local_max[2], local_max[3])
    return cute.arch.fmax(cute.arch.fmax(local_max[0], local_max[2]), init_val)


@cute.jit
def fadd_reduce(values: cute.TensorSSA, init_val: cutlass.Float32) -> cutlass.Float32:
    return values.reduce(cute.ReductionOp.ADD, init_val, 0)


@dataclass(frozen=True)
class FlashAttentionSoftmaxSM120:
    softmax_scale_log2: cutlass.Float32
    row_max: cute.Tensor
    row_sum: cute.Tensor

    @staticmethod
    def create(softmax_scale_log2: cutlass.Float32, num_rows: int) -> "FlashAttentionSoftmaxSM120":
        row_max = cute.make_rmem_tensor(num_rows, cutlass.Float32)
        row_sum = cute.make_rmem_tensor(num_rows, cutlass.Float32)
        row_max.fill(-cutlass.Float32.inf)
        row_sum.fill(0.0)
        return FlashAttentionSoftmaxSM120(softmax_scale_log2, row_max, row_sum)

    @cute.jit
    def online_softmax(
        self,
        acc_s: cute.Tensor,
        check_inf: cutlass.Constexpr = True,
    ) -> cute.Tensor:
        acc_s_mn = make_acc_tensor_mn_view(acc_s)
        row_scale = cute.make_fragment_like(self.row_max, cutlass.Float32)

        for row in cutlass.range_constexpr(cute.size(self.row_max)):
            scores = acc_s_mn[row, None].load()

            prev_row_max = self.row_max[row]
            curr_row_max = fmax_reduce(scores, prev_row_max)
            curr_row_max = cute.arch.warp_reduction_max(
                curr_row_max,
                threads_in_group=4,
            )

            self.row_max[row] = curr_row_max
            if cutlass.const_expr(check_inf):
                curr_row_max = 0.0 if curr_row_max == -cutlass.Float32.inf else curr_row_max
            correction = cute.math.exp2(
                (prev_row_max - curr_row_max) * self.softmax_scale_log2,
                fastmath=True,
            )
            probabilities = cute.math.exp2(
                scores * self.softmax_scale_log2 - curr_row_max * self.softmax_scale_log2,
                fastmath=True,
            )
            curr_row_sum = fadd_reduce(
                probabilities,
                self.row_sum[row] * correction,
            )

            row_scale[row] = correction
            self.row_sum[row] = curr_row_sum
            acc_s_mn[row, None].store(probabilities)

        return row_scale

    @cute.jit
    def rescale_O(self, acc_o: cute.Tensor, row_scale: cute.Tensor) -> None:
        acc_o_mn = make_acc_tensor_mn_view(acc_o)
        assert cute.size(row_scale) == cute.size(acc_o_mn, mode=[0])

        for row in cutlass.range_constexpr(cute.size(row_scale)):
            acc_o_mn[row, None].store(acc_o_mn[row, None].load() * row_scale[row])

    @cute.jit
    def normalize_output(self, acc_o: cute.Tensor) -> None:
        acc_o_mn = make_acc_tensor_mn_view(acc_o)

        for row in cutlass.range_constexpr(cute.size(self.row_sum)):
            total = cute.arch.warp_reduction_sum(self.row_sum[row], threads_in_group=4)
            invalid_total = total == 0.0 or total != total
            inverse_total = 1.0 if invalid_total else cute.arch.rcp_approx(total)
            acc_o_mn[row, None].store(acc_o_mn[row, None].load() * inverse_total)
