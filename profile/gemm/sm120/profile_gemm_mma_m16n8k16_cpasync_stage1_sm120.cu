#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cute/numeric/numeric_types.hpp>
#include <cute/tensor.hpp>

#include "gemm/sm120/gemm_mma_m16n8k16_cpasync_stage1_sm120.cuh"
#include "utils/utils.cuh"

namespace scinthil::profile::gemm::sm120 {

[[nodiscard]] bool profile_gemm_mma_m16n8k16_cpasync_stage1_sm120_fp16_fp16() {
  constexpr int m{8192};
  constexpr int n{8192};
  constexpr int k{16384};

  constexpr auto a_bytes = static_cast<std::size_t>(m * k * sizeof(cute::half_t));
  constexpr auto b_bytes = static_cast<std::size_t>(n * k * sizeof(cute::half_t));
  constexpr auto d_bytes = static_cast<std::size_t>(m * n * sizeof(cute::half_t));

  cute::half_t* a{nullptr};
  cute::half_t* b{nullptr};
  cute::half_t* d{nullptr};

  bool success{true};
  success = success && SCINTHIL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&a), a_bytes));
  success = success && SCINTHIL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&b), b_bytes));
  success = success && SCINTHIL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d), d_bytes));

  success = success && SCINTHIL_CUDA_CHECK(cudaMemset(a, 0, a_bytes));
  success = success && SCINTHIL_CUDA_CHECK(cudaMemset(b, 0, b_bytes));
  success = success && SCINTHIL_CUDA_CHECK(cudaMemset(d, 0, d_bytes));

  auto stride_a = cute::make_stride(k, cute::_1{});
  auto stride_b = cute::make_stride(k, cute::_1{});
  auto stride_d = cute::make_stride(n, cute::_1{});

  if (success) {
    const auto status =
        scinthil::gemm::sm120::launch_gemm_mma_m16n8k16_cpasync_stage1_sm120<cute::half_t, cute::half_t>(
            a, b, d, m, n, k, stride_a, stride_b, stride_d);
    success = SCINTHIL_CUDA_CHECK(status) && success;
  }
  success = success && SCINTHIL_CUDA_CHECK(cudaDeviceSynchronize());

  bool cleanup_success{true};
  if (a != nullptr) {
    cleanup_success = SCINTHIL_CUDA_CHECK(cudaFree(a)) && cleanup_success;
  }
  if (b != nullptr) {
    cleanup_success = SCINTHIL_CUDA_CHECK(cudaFree(b)) && cleanup_success;
  }
  if (d != nullptr) {
    cleanup_success = SCINTHIL_CUDA_CHECK(cudaFree(d)) && cleanup_success;
  }

  return success && cleanup_success;
}

[[nodiscard]] bool profile_gemm_mma_m16n8k16_cpasync_stage1_sm120_fp16_fp32() {
  constexpr int m{8192};
  constexpr int n{8192};
  constexpr int k{16384};

  constexpr auto a_bytes = static_cast<std::size_t>(m * k * sizeof(cute::half_t));
  constexpr auto b_bytes = static_cast<std::size_t>(n * k * sizeof(cute::half_t));
  constexpr auto d_bytes = static_cast<std::size_t>(m * n * sizeof(float));

  cute::half_t* a{nullptr};
  cute::half_t* b{nullptr};
  float* d{nullptr};

  bool success{true};
  success = success && SCINTHIL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&a), a_bytes));
  success = success && SCINTHIL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&b), b_bytes));
  success = success && SCINTHIL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d), d_bytes));

  success = success && SCINTHIL_CUDA_CHECK(cudaMemset(a, 0, a_bytes));
  success = success && SCINTHIL_CUDA_CHECK(cudaMemset(b, 0, b_bytes));
  success = success && SCINTHIL_CUDA_CHECK(cudaMemset(d, 0, d_bytes));

  auto stride_a = cute::make_stride(k, cute::_1{});
  auto stride_b = cute::make_stride(k, cute::_1{});
  auto stride_d = cute::make_stride(n, cute::_1{});

  if (success) {
    const auto status = scinthil::gemm::sm120::launch_gemm_mma_m16n8k16_cpasync_stage1_sm120<cute::half_t, float>(
        a, b, d, m, n, k, stride_a, stride_b, stride_d);
    success = SCINTHIL_CUDA_CHECK(status) && success;
  }
  success = success && SCINTHIL_CUDA_CHECK(cudaDeviceSynchronize());

  bool cleanup_success{true};
  if (a != nullptr) {
    cleanup_success = SCINTHIL_CUDA_CHECK(cudaFree(a)) && cleanup_success;
  }
  if (b != nullptr) {
    cleanup_success = SCINTHIL_CUDA_CHECK(cudaFree(b)) && cleanup_success;
  }
  if (d != nullptr) {
    cleanup_success = SCINTHIL_CUDA_CHECK(cudaFree(d)) && cleanup_success;
  }

  return success && cleanup_success;
}

}  // namespace scinthil::profile::gemm::sm120

int main() {
  using scinthil::profile::gemm::sm120::profile_gemm_mma_m16n8k16_cpasync_stage1_sm120_fp16_fp16;
  using scinthil::profile::gemm::sm120::profile_gemm_mma_m16n8k16_cpasync_stage1_sm120_fp16_fp32;

  const bool fp16_success = profile_gemm_mma_m16n8k16_cpasync_stage1_sm120_fp16_fp16();
  const bool fp32_success = profile_gemm_mma_m16n8k16_cpasync_stage1_sm120_fp16_fp32();
  const bool success = fp16_success && fp32_success;
  if (success) {
    std::printf("SM120 gemm_mma_m16n8k16_cpasync_stage1 profile launched FP16 and FP32 output variants\n");
    return EXIT_SUCCESS;
  }

  std::fprintf(stderr, "SM120 gemm_mma_m16n8k16_cpasync_stage1 profile failed\n");
  return EXIT_FAILURE;
}
