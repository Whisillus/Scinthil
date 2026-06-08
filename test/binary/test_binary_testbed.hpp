#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstddef>
#include <random>

#include "cuda/binary/binary.cuh"
#include "util/utils.cuh"

namespace scinthil::test::binary {

template <typename Dtype, typename Op>
void test_binary(std::size_t n = 1024) {
  thrust::host_vector<Dtype> lhs(n);
  thrust::host_vector<Dtype> rhs(n);
  std::mt19937 generator{0x5EEDU};
  std::uniform_real_distribution<float> distribution{-1.0F, 1.0F};

  for (std::size_t i{0}; i < n; ++i) {
    lhs[i] = static_cast<Dtype>(distribution(generator));
    rhs[i] = static_cast<Dtype>(distribution(generator));
  }

  thrust::device_vector<Dtype> d_lhs(lhs);
  thrust::device_vector<Dtype> d_rhs(rhs);
  thrust::device_vector<Dtype> d_out(n);

  Op op{};
  ASSERT_TRUE(SCINTHIL_CUDA_CHECK(
      ::scinthil::binary::launch_binary(d_lhs.data().get(), d_rhs.data().get(), d_out.data().get(), n, op)));
  ASSERT_TRUE(SCINTHIL_CUDA_CHECK(cudaDeviceSynchronize()));

  thrust::host_vector<Dtype> out(d_out);
  for (std::size_t i{0}; i < n; ++i) {
    EXPECT_EQ(out[i], op(lhs[i], rhs[i])) << "index: " << i;
  }
}

}  // namespace scinthil::test::binary
