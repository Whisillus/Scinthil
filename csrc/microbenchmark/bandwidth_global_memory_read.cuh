#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include "bandwidth_utils.cuh"
#include "ptx.cuh"

namespace scinthil::microbenchmark {

template <typename T, typename Index>
__global__ void bandwidth_global_memory_read_kernel(const T* input, Index iteration_count) {
  const Index thread_index = static_cast<Index>(blockIdx.x) * static_cast<Index>(blockDim.x) + threadIdx.x;
  const Index thread_count = static_cast<Index>(gridDim.x) * static_cast<Index>(blockDim.x);
  const Index iteration_stride = static_cast<Index>(GlobalMemoryAccessesPerThreadIteration) * thread_count;

#pragma unroll 1
  for (Index iteration{0}; iteration < iteration_count; ++iteration) {
    const Index index = iteration * iteration_stride + thread_index;
    ptx::ldg_128bit_discard_result(input + index);
    ptx::ldg_128bit_discard_result(input + index + thread_count);
    ptx::ldg_128bit_discard_result(input + index + 2U * thread_count);
    ptx::ldg_128bit_discard_result(input + index + 3U * thread_count);
  }
}

template <typename T, typename Index>
[[nodiscard]] bool run_bandwidth_global_memory_read_impl(const RunMicrobenchmarkOptions& options,
                                                         const cudaDeviceProp& properties,
                                                         BandwidthResources& resources, Index element_count) {
  auto* input = static_cast<T*>(resources.input);

  dim3 threads{};
  dim3 blocks{};
  if (!prepare_global_memory_launch(bandwidth_global_memory_read_kernel<T, Index>,
                                    static_cast<std::size_t>(element_count), properties, threads, blocks)) {
    return false;
  }
  const std::size_t element_count_size = static_cast<std::size_t>(element_count);
  const std::size_t grid_thread_count = static_cast<std::size_t>(blocks.x) * threads.x;
  const std::size_t iteration_stride = GlobalMemoryAccessesPerThreadIteration * grid_thread_count;
  assert(element_count_size % iteration_stride == 0);
  const Index iteration_count = static_cast<Index>(element_count_size / iteration_stride);

  const std::uint64_t single_iter_read_byte = element_count_size * sizeof(T);
  BenchmarkResult result{};
  if (!measure(
          options, resources,
          [=, &resources](std::size_t launch_index) {
            const std::size_t workspace_index = launch_index % resources.workspace_count;
            bandwidth_global_memory_read_kernel<T, Index><<<blocks, threads, 0, resources.stream>>>(
                input + workspace_index * element_count_size, iteration_count);
            return SCINTHIL_CUDA_CHECK(cudaGetLastError());
          },
          single_iter_read_byte, 0U, result)) {
    return false;
  }

  print_bandwidth_benchmark_result("global-memory read-only", options, properties, resources.workspace_count, blocks.x,
                                   threads.x, sizeof(Index) * 8U, result);
  return true;
}

[[nodiscard]] inline bool run_bandwidth_global_memory_read(const RunMicrobenchmarkOptions& options,
                                                           const cudaDeviceProp& properties,
                                                           BandwidthResources& resources) {
  using T = uint4;
  const std::size_t bytes = options.global_mib_per_buffer * Byte2MByte;
  const std::size_t element_count = bytes / sizeof(T);
  if (element_count <= std::numeric_limits<unsigned int>::max()) {
    return run_bandwidth_global_memory_read_impl<T, unsigned int>(options, properties, resources,
                                                                  static_cast<unsigned int>(element_count));
  }
  return run_bandwidth_global_memory_read_impl<T, std::size_t>(options, properties, resources, element_count);
}

}  // namespace scinthil::microbenchmark
