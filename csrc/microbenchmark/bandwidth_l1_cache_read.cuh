#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>

#include "bandwidth_utils.cuh"
#include "ptx.cuh"

namespace scinthil::microbenchmark {

template <typename T, unsigned int Unroll>
__global__ __launch_bounds__(256) void bandwidth_l1_cache_read_kernel(const T* input, T* output,
                                                                      unsigned int elements_per_block, int sweeps) {
  static_assert(Unroll >= 1U && Unroll <= 4U);
  const T* block_input = input + static_cast<std::size_t>(blockIdx.x) * elements_per_block;
  const unsigned int stride = blockDim.x;
  T prime_value = make_uint4(0U, 0U, 0U, 0U);
  T last_values[Unroll]{};

  // Prime this CTA's private slice here because a prior launch may schedule its CTA on another SM.
  for (unsigned int index = threadIdx.x; index < elements_per_block; index += stride) {
    prime_value = ptx::ldg_ca_128bit<T>(block_input + index);
  }
  __syncthreads();

  for (int sweep{0}; sweep < sweeps; ++sweep) {
    for (unsigned int index = threadIdx.x; index < elements_per_block; index += Unroll * stride) {
#pragma unroll
      for (unsigned int offset{0}; offset < Unroll; ++offset) {
        last_values[offset] = ptx::ldg_ca_128bit<T>(block_input + index + offset * stride);
      }
    }
  }

  T sum = prime_value;
#pragma unroll
  for (unsigned int offset{0}; offset < Unroll; ++offset) {
    sum.x += last_values[offset].x;
    sum.y += last_values[offset].y;
    sum.z += last_values[offset].z;
    sum.w += last_values[offset].w;
  }
  const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  ptx::stg_128bit(output + thread_index, sum);
}

template <unsigned int Unroll>
[[nodiscard]] bool run_bandwidth_l1_cache_read_impl(const RunMicrobenchmarkOptions& options,
                                                    const cudaDeviceProp& properties, BandwidthResources* resources,
                                                    unsigned int blocks_per_sm, std::size_t bytes_per_block) {
  using T = uint4;
  const dim3 threads{256U, 1U, 1U};

  if (!SCINTHIL_CUDA_CHECK(cudaFuncSetCacheConfig(bandwidth_l1_cache_read_kernel<T, Unroll>, cudaFuncCachePreferL1))) {
    return false;
  }
  const unsigned int block_count = blocks_per_sm * static_cast<unsigned int>(properties.multiProcessorCount);
  const std::size_t working_set_bytes = bytes_per_block * block_count;
  if (!resources->allocate_input(working_set_bytes)) {
    return false;
  }
  if (!resources->allocate_output(static_cast<std::size_t>(block_count) * threads.x * sizeof(T))) {
    return false;
  }

  auto* input = static_cast<T*>(resources->input);
  auto* output = static_cast<T*>(resources->output);
  const unsigned int elements_per_block = static_cast<unsigned int>(bytes_per_block / sizeof(T));
  const dim3 blocks{block_count, 1U, 1U};
  assert(elements_per_block % (Unroll * threads.x) == 0);

  float elapsed_ms{0.0F};
  if (!measure(
          options, resources,
          [=](std::size_t) {
            bandwidth_l1_cache_read_kernel<T, Unroll>
                <<<blocks, threads, 0, resources->stream>>>(input, output, elements_per_block, options.sweeps);
            return SCINTHIL_CUDA_CHECK(cudaGetLastError());
          },
          &elapsed_ms)) {
    return false;
  }

  print_l1_cache_bandwidth_benchmark_result(options, properties, blocks_per_sm, bytes_per_block, elapsed_ms);
  return true;
}

[[nodiscard]] inline bool run_bandwidth_l1_cache_read(const RunMicrobenchmarkOptions& options,
                                                      const cudaDeviceProp& properties, BandwidthResources* resources) {
  using T = uint4;
  const dim3 threads{256U, 1U, 1U};
  const std::size_t requested_bytes_per_sm = options.size_kib * Byte2KByte;
  const std::size_t bytes_per_thread_pass = static_cast<std::size_t>(threads.x) * sizeof(T);

  int maximum_blocks_per_sm{0};
  if (!SCINTHIL_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &maximum_blocks_per_sm, bandwidth_l1_cache_read_kernel<T, 4U>, threads.x, 0))) {
    return false;
  }

  unsigned int blocks_per_sm = static_cast<unsigned int>(maximum_blocks_per_sm);
  const std::size_t maximum_minimum_footprint = blocks_per_sm * bytes_per_thread_pass;
  if (requested_bytes_per_sm < maximum_minimum_footprint) {
    blocks_per_sm = static_cast<unsigned int>(requested_bytes_per_sm / bytes_per_thread_pass);
  }
  const std::size_t bytes_per_block =
      (requested_bytes_per_sm / blocks_per_sm / bytes_per_thread_pass) * bytes_per_thread_pass;
  const unsigned int loads_per_thread = static_cast<unsigned int>(bytes_per_block / bytes_per_thread_pass);
  if (loads_per_thread % 4U == 0) {
    return run_bandwidth_l1_cache_read_impl<4U>(options, properties, resources, blocks_per_sm, bytes_per_block);
  }
  if (loads_per_thread % 3U == 0) {
    return run_bandwidth_l1_cache_read_impl<3U>(options, properties, resources, blocks_per_sm, bytes_per_block);
  }
  if (loads_per_thread % 2U == 0) {
    return run_bandwidth_l1_cache_read_impl<2U>(options, properties, resources, blocks_per_sm, bytes_per_block);
  }
  return run_bandwidth_l1_cache_read_impl<1U>(options, properties, resources, blocks_per_sm, bytes_per_block);
}

}  // namespace scinthil::microbenchmark
