#pragma once

#include <cuda_runtime.h>

#include <cutlass/fast_math.h>

#include <cstddef>
#include <limits>

namespace scinthil::binary {

template <typename Dtype, typename Op>
__global__ void binary_kernel(const Dtype *lhs, const Dtype *rhs, Dtype *out,
                              std::size_t n, Op op) {
  const auto idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = op(lhs[idx], rhs[idx]);
  }
}

template <typename Dtype, typename Op>
[[nodiscard]]
cudaError_t launch_binary(const Dtype *lhs, const Dtype *rhs, Dtype *out,
                          std::size_t n, Op op, cudaStream_t stream = nullptr) {
  if (n == 0) {
    return cudaSuccess;
  }

  constexpr int threads_per_block{256};
  if (n > static_cast<std::size_t>(std::numeric_limits<int>::max() -
                                   (threads_per_block - 1))) {
    return cudaErrorInvalidValue;
  }

  const auto grid_size =
      cutlass::ceil_div(static_cast<int>(n), threads_per_block);

  binary_kernel<Dtype, Op>
      <<<grid_size, threads_per_block, 0, stream>>>(lhs, rhs, out, n, op);
  return cudaGetLastError();
}

} // namespace scinthil::binary
