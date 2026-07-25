#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdio>

#include "bandwidth_utils.cuh"
#include "ptx.cuh"

namespace scinthil::microbenchmark {

template <typename T>
__global__ __launch_bounds__(256) void bandwidth_l2_cache_read_kernel(const T* input, T* output,
                                                                      unsigned int element_count,
                                                                      int passes_per_launch) {
  const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int thread_count = gridDim.x * blockDim.x;
  T last_value0 = make_uint4(0U, 0U, 0U, 0U);
  T last_value1 = make_uint4(0U, 0U, 0U, 0U);
  T last_value2 = make_uint4(0U, 0U, 0U, 0U);
  T last_value3 = make_uint4(0U, 0U, 0U, 0U);

  for (int pass{0}; pass < passes_per_launch; ++pass) {
    for (unsigned int index = thread_index; index < element_count; index += 4U * thread_count) {
      last_value0 = ptx::ldg_cg_128bit<T>(input + index);
      last_value1 = ptx::ldg_cg_128bit<T>(input + index + thread_count);
      last_value2 = ptx::ldg_cg_128bit<T>(input + index + 2U * thread_count);
      last_value3 = ptx::ldg_cg_128bit<T>(input + index + 3U * thread_count);
    }
  }
  T sum = last_value0;
  sum.x += last_value1.x + last_value2.x + last_value3.x;
  sum.y += last_value1.y + last_value2.y + last_value3.y;
  sum.z += last_value1.z + last_value2.z + last_value3.z;
  sum.w += last_value1.w + last_value2.w + last_value3.w;
  ptx::stg_128bit(output + thread_index, sum);
}

[[nodiscard]] inline bool run_bandwidth_l2_cache_read(const RunMicrobenchmarkOptions& options,
                                                      const cudaDeviceProp& properties, BandwidthResources& resources) {
  using T = uint4;
  const std::size_t l2_cache_bytes = static_cast<std::size_t>(properties.l2CacheSize);
  std::size_t working_set_bytes =
      options.l2_working_set_mib == 0 ? l2_cache_bytes / 2U : options.l2_working_set_mib * Byte2MByte;
  if (working_set_bytes > l2_cache_bytes) {
    std::fprintf(stderr, "requested L2 working set exceeds the reported L2 capacity (%zu bytes)\n", l2_cache_bytes);
    return false;
  }

  const dim3 threads{256U, 1U, 1U};
  int blocks_per_sm{0};
  if (!SCINTHIL_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm, bandwidth_l2_cache_read_kernel<T>, threads.x, 0))) {
    return false;
  }

  const unsigned int maximum_block_count =
      static_cast<unsigned int>(blocks_per_sm) * static_cast<unsigned int>(properties.multiProcessorCount);
  const std::size_t bytes_per_block_iteration = 4U * threads.x * sizeof(T);
  const std::size_t size_limited_block_count = working_set_bytes / bytes_per_block_iteration;
  const unsigned int block_count = size_limited_block_count < maximum_block_count
                                       ? static_cast<unsigned int>(size_limited_block_count)
                                       : maximum_block_count;
  const std::size_t bytes_per_grid_iteration = bytes_per_block_iteration * block_count;
  working_set_bytes = working_set_bytes / bytes_per_grid_iteration * bytes_per_grid_iteration;
  if (!resources.allocate_input(working_set_bytes)) {
    return false;
  }
  if (!resources.allocate_output(static_cast<std::size_t>(block_count) * threads.x * sizeof(T))) {
    return false;
  }

  auto* input = static_cast<T*>(resources.input);
  auto* output = static_cast<T*>(resources.output);
  const unsigned int element_count = static_cast<unsigned int>(working_set_bytes / sizeof(T));
  const dim3 blocks{block_count, 1U, 1U};
  assert(element_count % (4U * block_count * threads.x) == 0);

  float elapsed_ms{0.0F};
  if (!measure(
          options, resources,
          [=, &resources](std::size_t) {
            bandwidth_l2_cache_read_kernel<T>
                <<<blocks, threads, 0, resources.stream>>>(input, output, element_count, options.passes_per_launch);
            return SCINTHIL_CUDA_CHECK(cudaGetLastError());
          },
          elapsed_ms)) {
    return false;
  }

  print_l2_cache_bandwidth_benchmark_result(options, properties, working_set_bytes, block_count, elapsed_ms);
  return true;
}

}  // namespace scinthil::microbenchmark
