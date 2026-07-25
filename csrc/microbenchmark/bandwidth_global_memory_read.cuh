#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>

#include "bandwidth_utils.cuh"

namespace scinthil::microbenchmark {

template <typename T>
__global__ void bandwidth_global_memory_read_kernel(const volatile T* input, std::size_t element_count) {
  const std::size_t thread_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t thread_count = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index = thread_index; index < element_count; index += thread_count) {
    (void)input[index];
  }
}

[[nodiscard]] inline bool run_bandwidth_global_memory_read(const RunMicrobenchmarkOptions& options,
                                                           const cudaDeviceProp& properties,
                                                           BandwidthResources* resources) {
  const std::size_t bytes = options.size_mib * Byte2MByte;
  const std::size_t element_count = bytes / sizeof(unsigned int);
  auto* input = static_cast<unsigned int*>(resources->input);

  int minimum_num_blocks{0};
  int num_threads_x{0};
  if (!SCINTHIL_CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
          &minimum_num_blocks, &num_threads_x, bandwidth_global_memory_read_kernel<unsigned int>, 0, 0))) {
    return false;
  }
  if (minimum_num_blocks <= 0 || num_threads_x <= 0) {
    std::fprintf(stderr, "could not determine a valid global-memory read launch size\n");
    return false;
  }
  const dim3 threads{static_cast<unsigned int>(num_threads_x), 1, 1};
  const std::size_t num_blocks_x =
      (element_count + static_cast<std::size_t>(threads.x) - 1) / static_cast<std::size_t>(threads.x);
  if (num_blocks_x > static_cast<std::size_t>(properties.maxGridSize[0])) {
    std::fprintf(stderr, "global-memory read grid exceeds the device limit\n");
    return false;
  }
  const dim3 blocks{static_cast<unsigned int>(num_blocks_x), 1, 1};

  float elapsed_ms{0.0F};
  if (!measure(
          options, resources,
          [=] {
            bandwidth_global_memory_read_kernel<unsigned int>
                <<<blocks, threads, 0, resources->stream>>>(input, element_count);
            return SCINTHIL_CUDA_CHECK(cudaGetLastError());
          },
          &elapsed_ms)) {
    return false;
  }

  print_bandwidth_benchmark_result("global-memory read-only", options, properties, elapsed_ms, 1.0);
  return true;
}

}  // namespace scinthil::microbenchmark
