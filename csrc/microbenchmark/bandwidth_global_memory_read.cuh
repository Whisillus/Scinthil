#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>

#include "bandwidth_utils.cuh"
#include "ptx.cuh"

namespace scinthil::microbenchmark {

template <typename T>
__global__ void bandwidth_global_memory_read_kernel(const T* input, std::size_t element_count) {
  const std::size_t thread_index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t thread_count = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index = thread_index; index < element_count; index += thread_count) {
    ptx::ldg_128bit_discard_result(input + index);
  }
}

[[nodiscard]] inline bool run_bandwidth_global_memory_read(const RunMicrobenchmarkOptions& options,
                                                           const cudaDeviceProp& properties,
                                                           BandwidthResources& resources) {
  using T = uint4;
  const std::size_t bytes = options.global_mib_per_buffer * Byte2MByte;
  const std::size_t element_count = bytes / sizeof(T);
  auto* input = static_cast<T*>(resources.input);

  int minimum_grid_size{0};
  int num_threads_x{0};
  if (!SCINTHIL_CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minimum_grid_size, &num_threads_x,
                                                              bandwidth_global_memory_read_kernel<T>, 0, 0))) {
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
          [=, &resources](std::size_t launch_index) {
            const std::size_t workspace_index = launch_index % resources.workspace_count;
            bandwidth_global_memory_read_kernel<T>
                <<<blocks, threads, 0, resources.stream>>>(input + workspace_index * element_count, element_count);
            return SCINTHIL_CUDA_CHECK(cudaGetLastError());
          },
          elapsed_ms)) {
    return false;
  }

  print_bandwidth_benchmark_result("global-memory read-only", options, properties, resources.workspace_count,
                                   elapsed_ms, 1.0);
  return true;
}

}  // namespace scinthil::microbenchmark
