import cutlass
import cutlass.cute as cute


@cute.jit
def get_predicate_load_q_seqlen(
    tQcQ: cute.Tensor,
    tile_m_idx: cutlass.Int32,
    seqlen_q: cutlass.Int32,
    tile_m: cutlass.Constexpr,
) -> cute.Tensor:
    q_pred_layout = cute.make_layout(
        (
            cute.size(tQcQ, mode=[0, 1]),
            cute.size(tQcQ, mode=[1]),
            cute.size(tQcQ, mode=[2]),
        ),
        stride=(0, 1, 0),
    )
    tQpQ = cute.make_rmem_tensor(q_pred_layout, cutlass.Boolean)
    for q_row in cutlass.range_constexpr(tQpQ.shape[1]):
        tQpQ[0, q_row, 0] = cute.elem_less(
            tile_m_idx * tile_m + tQcQ[(0, 0), q_row, 0][0],
            seqlen_q,
        )
    return tQpQ


@cute.jit
def get_predicate_load_q_seqlen_headdim(
    tQcQ: cute.Tensor,
    tile_m_idx: cutlass.Int32,
    seqlen_q: cutlass.Int32,
    headdim_qk: cutlass.Int32,
    tile_m: cutlass.Constexpr,
) -> cute.Tensor:
    q_pred_shape = (
        cute.size(tQcQ, mode=[0, 1]),
        cute.size(tQcQ, mode=[1]),
        cute.size(tQcQ, mode=[2]),
    )
    q_pred_stride = (
        1,
        q_pred_shape[0],
        q_pred_shape[0] * q_pred_shape[1],
    )
    q_pred_layout = cute.make_layout(q_pred_shape, stride=q_pred_stride)
    tQpQ = cute.make_rmem_tensor(q_pred_layout, cutlass.Boolean)
    for atom_rest in cutlass.range_constexpr(tQpQ.shape[0]):
        for q_row in cutlass.range_constexpr(tQpQ.shape[1]):
            for q_k in cutlass.range_constexpr(tQpQ.shape[2]):
                q_coord = tQcQ[(0, atom_rest), q_row, q_k]
                tQpQ[atom_rest, q_row, q_k] = cute.elem_less(
                    (tile_m_idx * tile_m + q_coord[0], q_coord[1]),
                    (seqlen_q, headdim_qk),
                )
    return tQpQ


def make_acc_tensor_mn_view(acc: cute.Tensor) -> cute.Tensor:
    compact_shape = acc.layout.shape
    # CuTe's default compact layout supplies the canonical strides used to
    # regroup an MMA accumulator into logical M and N modes.
    compact_layout = cute.make_layout(compact_shape)
    compact_stride = compact_layout.stride
    mn_shape = (
        (compact_shape[0][1], compact_shape[1]),
        (compact_shape[0][0], compact_shape[2]),
    )
    mn_stride = (
        (compact_stride[0][1], compact_stride[1]),
        (compact_stride[0][0], compact_stride[2]),
    )
    mn_layout = cute.make_layout(mn_shape, stride=mn_stride)
    return cute.make_tensor(acc.iterator, cute.composition(acc.layout, mn_layout))
