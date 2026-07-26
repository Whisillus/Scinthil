#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>

#include "bandwidth_utils.cuh"
#include "ptx.cuh"

namespace scinthil::microbenchmark {

__global__ void bandwidth_tma_bulk_write_kernel(unsigned char* output, std::size_t tile_count,
                                                unsigned int transfer_bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  extern __shared__ __align__(128) unsigned char shared_storage[];
  for (std::size_t tile_index = blockIdx.x; tile_index < tile_count; tile_index += gridDim.x) {
    if (ptx::elect_one_sync()) {
      ptx::tma_bulk_shared_to_global(output + tile_index * transfer_bytes, shared_storage, transfer_bytes);
      ptx::tma_store_commit();
    }
  }
  ptx::tma_store_wait();
#else
  static_cast<void>(output);
  static_cast<void>(tile_count);
  static_cast<void>(transfer_bytes);
#endif
}

[[nodiscard]] inline bool run_bandwidth_tma_bulk_write(const RunMicrobenchmarkOptions& options,
                                                       const cudaDeviceProp& properties,
                                                       BandwidthResources& resources) {
  if (properties.major < 9) {
    std::fprintf(stderr, "TMA benchmarks require compute capability 9.0 or newer\n");
    return false;
  }

  const std::size_t buffer_bytes = options.global_mib_per_buffer * Byte2MByte;
  const std::size_t transfer_bytes = options.shared_kib_per_block * Byte2KByte;
  if (buffer_bytes % transfer_bytes != 0) {
    std::fprintf(stderr, "TMA buffer size must be divisible by the shared-memory transfer size\n");
    return false;
  }
  const std::size_t tile_count = buffer_bytes / transfer_bytes;
  const std::size_t dynamic_shared_bytes = transfer_bytes;
  if (dynamic_shared_bytes > properties.sharedMemPerBlockOptin) {
    std::fprintf(stderr, "TMA shared-memory allocation exceeds the device per-block limit\n");
    return false;
  }
  if (!SCINTHIL_CUDA_CHECK(cudaFuncSetAttribute(bandwidth_tma_bulk_write_kernel,
                                                cudaFuncAttributePreferredSharedMemoryCarveout,
                                                cudaSharedmemCarveoutMaxShared))) {
    return false;
  }
  if (dynamic_shared_bytes > properties.sharedMemPerBlock &&
      !SCINTHIL_CUDA_CHECK(cudaFuncSetAttribute(bandwidth_tma_bulk_write_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                static_cast<int>(dynamic_shared_bytes)))) {
    return false;
  }

  const dim3 blocks{static_cast<unsigned int>(properties.multiProcessorCount), 1U, 1U};
  const dim3 threads{32U, 1U, 1U};
  auto* output = static_cast<unsigned char*>(resources.output);
  float elapsed_ms{0.0F};
  if (!measure(
          options, resources,
          [=, &resources](std::size_t launch_index) {
            const std::size_t workspace_index = launch_index % resources.workspace_count;
            bandwidth_tma_bulk_write_kernel<<<blocks, threads, dynamic_shared_bytes, resources.stream>>>(
                output + workspace_index * buffer_bytes, tile_count, static_cast<unsigned int>(transfer_bytes));
            return SCINTHIL_CUDA_CHECK(cudaGetLastError());
          },
          elapsed_ms)) {
    return false;
  }

  print_tma_bandwidth_benchmark_result("TMA bulk write-only", options, properties, resources.workspace_count,
                                       elapsed_ms, 1.0);
  return true;
}

}  // namespace scinthil::microbenchmark
