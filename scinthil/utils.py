from dataclasses import dataclass
from typing import Any

import cuda.bindings.runtime as cuda_runtime
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


@dataclass(frozen=True)
class TensorLayout:
    """Host representation of a flat affine tensor layout."""

    shape: tuple[int, ...]
    stride: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.shape or len(self.shape) != len(self.stride):
            raise ValueError("shape and stride must have the same positive rank")
        if any(extent <= 0 for extent in self.shape):
            raise ValueError("shape extents must be positive")
        if any(value < 0 for value in self.stride):
            raise ValueError("negative strides are not supported")

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def cosize(self) -> int:
        return 1 + sum((extent - 1) * stride for extent, stride in zip(self.shape, self.stride, strict=True))


class CudaTensor:
    """Own a cudaMalloc allocation and its host tensor layout."""

    def __init__(
        self,
        ptr: cute.Pointer,
        layout: TensorLayout,
        address: int | None,
    ) -> None:
        self.ptr = ptr
        self.layout = layout
        self._address = address
        self._freed = False

    def to_tensor(self, *, loc: Any = None, ip: Any = None) -> cute.Tensor:
        layout = cute.make_layout(self.layout.shape, stride=self.layout.stride, loc=loc, ip=ip)
        return cute.make_tensor(self.ptr, layout, loc=loc, ip=ip)

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


__all__ = ["CudaTensor", "TensorLayout", "check_cuda_runtime"]
