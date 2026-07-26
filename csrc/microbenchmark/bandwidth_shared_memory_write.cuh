#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "bandwidth_utils.cuh"
#include "ptx.cuh"

namespace scinthil::microbenchmark {

__global__ __launch_bounds__(256) void bandwidth_shared_memory_write_kernel(unsigned int element_count,
                                                                            int passes_per_launch) {
  using T = uint4;
  extern __shared__ __align__(16) unsigned char shared_storage[];
  const unsigned int shared_address = ptx::cvta_to_shared(shared_storage);
  constexpr unsigned int element_bytes = sizeof(T);

  const unsigned int thread_value = blockIdx.x * blockDim.x + threadIdx.x;
  T value = make_uint4(thread_value, thread_value + 1U, thread_value + 2U, thread_value + 3U);
  const unsigned int stride = blockDim.x;
  for (unsigned int index = threadIdx.x; index < element_count; index += 4U * stride) {
    const unsigned int address0 = shared_address + index * element_bytes;
    const unsigned int address1 = shared_address + (index + stride) * element_bytes;
    const unsigned int address2 = shared_address + (index + 2U * stride) * element_bytes;
    const unsigned int address3 = shared_address + (index + 3U * stride) * element_bytes;
#pragma unroll 1
    for (int pass{0}; pass < passes_per_launch; pass += SharedMemoryPassesPerLaunchGranularity) {
#pragma unroll
      for (int pass_offset{0}; pass_offset < SharedMemoryPassesPerLaunchGranularity; ++pass_offset) {
        ptx::sts_128bit(address0, value);
        ptx::sts_128bit(address1, value);
        ptx::sts_128bit(address2, value);
        ptx::sts_128bit(address3, value);
      }
    }
  }
}

[[nodiscard]] inline bool run_bandwidth_shared_memory_write(const RunMicrobenchmarkOptions& options,
                                                            const cudaDeviceProp& properties,
                                                            BandwidthResources& resources) {
  dim3 blocks{};
  const dim3 threads{256U, 1U, 1U};
  if (!prepare_shared_memory_launch(bandwidth_shared_memory_write_kernel, options, properties, threads, blocks)) {
    return false;
  }
  const std::size_t shared_memory_bytes = options.shared_kib_per_block * Byte2KByte;
  const unsigned int element_count = static_cast<unsigned int>(shared_memory_bytes / sizeof(uint4));

  const std::uint64_t single_iter_write_byte =
      static_cast<std::uint64_t>(shared_memory_bytes) * blocks.x * options.passes_per_launch;
  BenchmarkResult result{};
  if (!measure(
          options, resources,
          [=, &resources](std::size_t) {
            bandwidth_shared_memory_write_kernel<<<blocks, threads, shared_memory_bytes, resources.stream>>>(
                element_count, options.passes_per_launch);
            return SCINTHIL_CUDA_CHECK(cudaGetLastError());
          },
          0U, single_iter_write_byte, result)) {
    return false;
  }

  print_shared_memory_bandwidth_benchmark_result("shared-memory write-only", options, properties, blocks.x, result);
  return true;
}

}  // namespace scinthil::microbenchmark
