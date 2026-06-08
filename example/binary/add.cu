#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda/binary/binary.cuh"
#include "util/utils.cuh"

namespace {

struct Add {
  __host__ __device__ constexpr float operator()(float lhs, float rhs) const { return lhs + rhs; }
};

}  // namespace

int main() {
  constexpr std::size_t n{1024};
  const auto bytes = n * sizeof(float);

  std::vector<float> lhs(n);
  std::vector<float> rhs(n);

  for (std::size_t i{0}; i < n; ++i) {
    lhs[i] = static_cast<float>(i) * 0.5F;
    rhs[i] = static_cast<float>(i) * 2.0F;
  }

  float* d_lhs{nullptr};
  float* d_rhs{nullptr};
  float* d_out{nullptr};

  bool success{true};
  success = success && SCINTHIL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_lhs), bytes));
  success = success && SCINTHIL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_rhs), bytes));
  success = success && SCINTHIL_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out), bytes));
  success = success && SCINTHIL_CUDA_CHECK(cudaMemcpy(d_lhs, lhs.data(), bytes, cudaMemcpyHostToDevice));
  success = success && SCINTHIL_CUDA_CHECK(cudaMemcpy(d_rhs, rhs.data(), bytes, cudaMemcpyHostToDevice));
  success = success && SCINTHIL_CUDA_CHECK(scinthil::binary::launch_binary(d_lhs, d_rhs, d_out, n, Add{}));
  success = success && SCINTHIL_CUDA_CHECK(cudaDeviceSynchronize());

  bool cleanup_success{true};
  if (d_lhs != nullptr) {
    cleanup_success = SCINTHIL_CUDA_CHECK(cudaFree(d_lhs)) && cleanup_success;
  }
  if (d_rhs != nullptr) {
    cleanup_success = SCINTHIL_CUDA_CHECK(cudaFree(d_rhs)) && cleanup_success;
  }
  if (d_out != nullptr) {
    cleanup_success = SCINTHIL_CUDA_CHECK(cudaFree(d_out)) && cleanup_success;
  }

  success = success && cleanup_success;
  if (success) {
    std::printf("binary add example launched successfully for %zu elements\n", n);
    return EXIT_SUCCESS;
  }

  std::fprintf(stderr, "binary add example failed\n");
  return EXIT_FAILURE;
}
