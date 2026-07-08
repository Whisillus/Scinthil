#include <cuda_runtime.h>

#include <cstdlib>
#include <cute/numeric/numeric_types.hpp>
#include <cute/tensor.hpp>

#include "gemm/sm120/naive_gemm_sm120.cuh"

namespace scinthil::profile::gemm::sm120 {

namespace detail {

template <class ElementAB, class ElementD, class StrideA, class StrideB, class StrideD>
__launch_bounds__(
    scinthil::gemm::sm120::NaiveGEMMKernelSM120<ElementAB, ElementD, StrideA, StrideB, StrideD>::threads_per_block)
    __global__ void naive_gemm_kernel_trampoline(const ElementAB* a, const ElementAB* b, ElementD* d, int m, int n,
                                                 int k, StrideA stride_a, StrideB stride_b, StrideD stride_d) {
  using Kernel = scinthil::gemm::sm120::NaiveGEMMKernelSM120<ElementAB, ElementD, StrideA, StrideB, StrideD>;
  Kernel::run(a, b, d, m, n, k, stride_a, stride_b, stride_d);
}

}  // namespace detail

template <class ElementAB, class ElementD, class StrideA, class StrideB, class StrideD>
[[nodiscard]] cudaError_t launch_naive_gemm_sm120(const ElementAB* a, const ElementAB* b, ElementD* d, int m, int n,
                                                  int k, StrideA stride_a, StrideB stride_b, StrideD stride_d,
                                                  cudaStream_t stream = nullptr) {
  using Kernel = scinthil::gemm::sm120::NaiveGEMMKernelSM120<ElementAB, ElementD, StrideA, StrideB, StrideD>;
  using TileShape = typename Kernel::TileShape;

  if (m <= 0 || n <= 0 || k <= 0) {
    return cudaErrorInvalidValue;
  }
  if (a == nullptr || b == nullptr || d == nullptr) {
    return cudaErrorInvalidValue;
  }
  if ((m % cute::size<0>(TileShape{})) != 0 || (n % cute::size<1>(TileShape{})) != 0 ||
      (k % cute::size<2>(TileShape{})) != 0) {
    return cudaErrorInvalidValue;
  }

  dim3 grid(static_cast<unsigned>(m / cute::size<0>(TileShape{})),
            static_cast<unsigned>(n / cute::size<1>(TileShape{})));
  detail::naive_gemm_kernel_trampoline<ElementAB, ElementD, StrideA, StrideB, StrideD>
      <<<grid, Kernel::threads_per_block, Kernel::smem_size, stream>>>(a, b, d, m, n, k, stride_a, stride_b, stride_d);
  return cudaGetLastError();
}

}  // namespace scinthil::profile::gemm::sm120

int main() {
  auto stride_a = cute::make_stride(16, cute::_1{});
  auto stride_b = cute::make_stride(16, cute::_1{});
  auto stride_d = cute::make_stride(8, cute::_1{});

  const auto bf16_status = scinthil::profile::gemm::sm120::launch_naive_gemm_sm120<cute::bfloat16_t, float>(
      nullptr, nullptr, nullptr, 0, 0, 0, stride_a, stride_b, stride_d);

  return bf16_status == cudaErrorInvalidValue ? EXIT_SUCCESS : EXIT_FAILURE;
}
