#include "cuda/binary/binary.cuh"
#include "ref.hpp"
#include "util/utils.cuh"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace {

struct Add {
  __host__ __device__ constexpr float operator()(float lhs, float rhs) const {
    return lhs + rhs;
  }
};

struct CudaFreeDeleter {
  void operator()(float *ptr) const {
    if (ptr != nullptr) {
      static_cast<void>(cudaFree(ptr));
    }
  }
};

using DeviceFloatPtr = std::unique_ptr<float, CudaFreeDeleter>;

} // namespace

TEST(ScinthilBinaryAdd, AddsFloatVectors) {
  constexpr std::size_t n{1024};
  constexpr float tolerance{1.0e-5F};
  const auto bytes = n * sizeof(float);

  std::vector<float> lhs(n);
  std::vector<float> rhs(n);
  std::vector<float> out(n, 0.0F);

  for (std::size_t i{0}; i < n; ++i) {
    lhs[i] = static_cast<float>(i) * 0.5F;
    rhs[i] = static_cast<float>(i) * 2.0F;
  }

  float *raw_lhs{nullptr};
  ASSERT_TRUE(SCINTHIL_CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&raw_lhs), bytes)));
  DeviceFloatPtr d_lhs{raw_lhs};

  float *raw_rhs{nullptr};
  ASSERT_TRUE(SCINTHIL_CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&raw_rhs), bytes)));
  DeviceFloatPtr d_rhs{raw_rhs};

  float *raw_out{nullptr};
  ASSERT_TRUE(SCINTHIL_CUDA_CHECK(
      cudaMalloc(reinterpret_cast<void **>(&raw_out), bytes)));
  DeviceFloatPtr d_out{raw_out};

  ASSERT_TRUE(SCINTHIL_CUDA_CHECK(
      cudaMemcpy(d_lhs.get(), lhs.data(), bytes, cudaMemcpyHostToDevice)));
  ASSERT_TRUE(SCINTHIL_CUDA_CHECK(
      cudaMemcpy(d_rhs.get(), rhs.data(), bytes, cudaMemcpyHostToDevice)));
  ASSERT_TRUE(SCINTHIL_CUDA_CHECK(scinthil::binary::launch_binary(
      d_lhs.get(), d_rhs.get(), d_out.get(), n, Add{})));
  ASSERT_TRUE(SCINTHIL_CUDA_CHECK(cudaDeviceSynchronize()));
  ASSERT_TRUE(SCINTHIL_CUDA_CHECK(
      cudaMemcpy(out.data(), d_out.get(), bytes, cudaMemcpyDeviceToHost)));

  EXPECT_TRUE(scinthil::test::binary::verify_add(
      std::span<const float>{lhs.data(), lhs.size()},
      std::span<const float>{rhs.data(), rhs.size()},
      std::span<const float>{out.data(), out.size()}, tolerance));
}
