from typing import Any

import cuda.bindings.runtime as cuda_runtime
import cutlass
import cutlass.cute as cute


def check_cuda_runtime(result: tuple[Any, ...]) -> Any:
    error, *values = result
    if error != cuda_runtime.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA Runtime call failed with {error}")
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return tuple(values)


class CudaTensor:
    """Own a cudaMalloc allocation and its CuTe layout."""

    def __init__(
        self,
        ptr: cute.Pointer,
        layout: cute.Layout | cute.ComposedLayout,
        address: int | None,
    ) -> None:
        self.ptr = ptr
        self.layout = layout
        self._address = address
        self._freed = False

    def to_tensor(self, *, loc: Any = None, ip: Any = None) -> cute.Tensor:
        return cute.make_tensor(self.ptr, self.layout, loc=loc, ip=ip)

    def free(self) -> None:
        if self._freed:
            return
        if self._address is None:
            raise RuntimeError("Only the host allocation owner can free a CUDA tensor")
        check_cuda_runtime(cuda_runtime.cudaFree(self._address))
        self._freed = True

    def __c_pointers__(self) -> Any:
        if self._freed:
            raise RuntimeError("Cannot use a freed CUDA tensor")
        return self.ptr.__c_pointers__()

    def __get_mlir_types__(self) -> Any:
        return self.ptr.__get_mlir_types__()

    def __extract_mlir_values__(self) -> Any:
        return self.ptr.__extract_mlir_values__()

    def __new_from_mlir_values__(self, values: Any) -> "CudaTensor":
        return CudaTensor(
            self.ptr.__new_from_mlir_values__(values),
            self.layout,
            address=None,
        )


def get_input_Q(
    *,
    q_layout: cute.Layout | cute.ComposedLayout,
    dtype: type[Any],
) -> tuple[CudaTensor, int, None, None]:
    """Allocate BSHD Q and return fixed-length sequence metadata."""
    if dtype not in (cutlass.Float16, cutlass.BFloat16):
        raise TypeError("FlashAttention inputs must use FP16 or BF16")
    if cute.rank(q_layout) != 4:
        raise ValueError("q_layout must describe a rank-4 BSHD tensor")

    num_bytes = int(cute.cosize(q_layout)) * dtype.width // 8
    if num_bytes <= 0:
        raise ValueError("q_layout must describe non-empty storage")
    address = check_cuda_runtime(cuda_runtime.cudaMalloc(num_bytes))
    ptr = cute.runtime.make_ptr(
        dtype,
        address,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )

    mQ = CudaTensor(ptr, q_layout, address)
    max_seqlen_q = int(cute.size(q_layout, mode=[1]))
    return mQ, max_seqlen_q, None, None


def get_input_KV(
    *,
    k_layout: cute.Layout | cute.ComposedLayout,
    v_layout: cute.Layout | cute.ComposedLayout,
    dtype: type[Any],
) -> tuple[CudaTensor, CudaTensor, int, None, None]:
    """Allocate BSHD K/V and return their shared sequence metadata."""
    if dtype not in (cutlass.Float16, cutlass.BFloat16):
        raise TypeError("FlashAttention inputs must use FP16 or BF16")
    if cute.rank(k_layout) != 4 or cute.rank(v_layout) != 4:
        raise ValueError("k_layout and v_layout must describe rank-4 BSHD tensors")
    for mode in (0, 1, 2):
        if cute.size(k_layout, mode=[mode]) != cute.size(v_layout, mode=[mode]):
            raise ValueError("K and V layouts must have matching batch, sequence, and head extents")

    k_num_bytes = int(cute.cosize(k_layout)) * dtype.width // 8
    v_num_bytes = int(cute.cosize(v_layout)) * dtype.width // 8
    if k_num_bytes <= 0 or v_num_bytes <= 0:
        raise ValueError("K/V layouts must describe non-empty storage")

    k_address = check_cuda_runtime(cuda_runtime.cudaMalloc(k_num_bytes))
    v_address = check_cuda_runtime(cuda_runtime.cudaMalloc(v_num_bytes))
    k_ptr = cute.runtime.make_ptr(
        dtype,
        k_address,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    v_ptr = cute.runtime.make_ptr(
        dtype,
        v_address,
        cute.AddressSpace.gmem,
        assumed_align=16,
    )

    mK = CudaTensor(k_ptr, k_layout, k_address)
    mV = CudaTensor(v_ptr, v_layout, v_address)
    max_seqlen_kv = int(cute.size(k_layout, mode=[1]))
    return mK, mV, max_seqlen_kv, None, None


__all__ = ["CudaTensor", "check_cuda_runtime", "get_input_KV", "get_input_Q"]
