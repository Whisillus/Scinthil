import cutlass.cute as cute


def make_acc_tensor_mn_view(acc: cute.Tensor) -> cute.Tensor:
    compact_shape = acc.layout.shape
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
