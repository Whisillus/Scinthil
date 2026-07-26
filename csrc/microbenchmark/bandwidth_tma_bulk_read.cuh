#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "bandwidth_utils.cuh"
#include "ptx.cuh"

namespace scinthil::microbenchmark {

template <unsigned int Stages>
__global__ void bandwidth_tma_bulk_read_kernel(const unsigned char* input, std::size_t tile_count,
                                               unsigned int transfer_bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  static_assert(Stages >= 1U && Stages <= TmaMaximumBenchmarkStages);
  extern __shared__ __align__(128) unsigned char shared_storage[];
  auto* barriers = reinterpret_cast<std::uint64_t*>(shared_storage + Stages * transfer_bytes);
  if (ptx::elect_one_sync()) {
#pragma unroll
    for (unsigned int stage_index{0}; stage_index < Stages; ++stage_index) {
      ptx::mbarrier_init(barriers + stage_index);
    }
    ptx::fence_mbarrier_init();
  }
  __syncwarp();

  unsigned int active_stage_mask{0};
  unsigned int phase_mask{0};
  unsigned int stage{0};
  for (std::size_t tile_index = blockIdx.x; tile_index < tile_count; tile_index += gridDim.x) {
    const unsigned int stage_bit = 1U << stage;
    if ((active_stage_mask & stage_bit) != 0U) {
      ptx::mbarrier_wait(barriers + stage, (phase_mask & stage_bit) != 0U);
      phase_mask ^= stage_bit;
    }
    if (ptx::elect_one_sync()) {
      auto* stage_buffer = shared_storage + stage * transfer_bytes;
      ptx::mbarrier_arrive_expect_tx(barriers + stage, transfer_bytes);
      ptx::tma_bulk_global_to_shared(input + tile_index * transfer_bytes, stage_buffer, barriers + stage,
                                     transfer_bytes);
    }
    active_stage_mask |= stage_bit;
    stage = stage + 1U == Stages ? 0U : stage + 1U;
  }

#pragma unroll
  for (unsigned int stage_index{0}; stage_index < Stages; ++stage_index) {
    const unsigned int stage_bit = 1U << stage_index;
    if ((active_stage_mask & stage_bit) != 0U) {
      ptx::mbarrier_wait(barriers + stage_index, (phase_mask & stage_bit) != 0U);
    }
  }
#else
  static_cast<void>(input);
  static_cast<void>(tile_count);
  static_cast<void>(transfer_bytes);
#endif
}

template <unsigned int Stages>
[[nodiscard]] bool run_bandwidth_tma_bulk_read_impl(const RunMicrobenchmarkOptions& options,
                                                    const cudaDeviceProp& properties, BandwidthResources& resources,
                                                    std::size_t buffer_bytes, std::size_t tile_count,
                                                    std::size_t dynamic_shared_bytes) {
  const std::size_t transfer_bytes = options.shared_kib_per_block * Byte2KByte;
  if (!SCINTHIL_CUDA_CHECK(cudaFuncSetAttribute(bandwidth_tma_bulk_read_kernel<Stages>,
                                                cudaFuncAttributePreferredSharedMemoryCarveout,
                                                cudaSharedmemCarveoutMaxShared))) {
    return false;
  }
  if (dynamic_shared_bytes > properties.sharedMemPerBlock &&
      !SCINTHIL_CUDA_CHECK(cudaFuncSetAttribute(bandwidth_tma_bulk_read_kernel<Stages>,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                static_cast<int>(dynamic_shared_bytes)))) {
    return false;
  }

  const dim3 blocks{static_cast<unsigned int>(properties.multiProcessorCount), 1U, 1U};
  const dim3 threads{32U, 1U, 1U};
  auto* input = static_cast<unsigned char*>(resources.input);
  BenchmarkResult result{};
  if (!measure(
          options, resources,
          [=, &resources](std::size_t launch_index) {
            const std::size_t workspace_index = launch_index % resources.workspace_count;
            bandwidth_tma_bulk_read_kernel<Stages><<<blocks, threads, dynamic_shared_bytes, resources.stream>>>(
                input + workspace_index * buffer_bytes, tile_count, static_cast<unsigned int>(transfer_bytes));
            return SCINTHIL_CUDA_CHECK(cudaGetLastError());
          },
          buffer_bytes, 0U, result)) {
    return false;
  }

  print_tma_bandwidth_benchmark_result("TMA bulk read-only", options, properties, resources.workspace_count, Stages,
                                       result);
  return true;
}

[[nodiscard]] inline bool run_bandwidth_tma_bulk_read(const RunMicrobenchmarkOptions& options,
                                                      const cudaDeviceProp& properties, BandwidthResources& resources) {
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
  const unsigned int stage_count = options.tma_benchmark_stages;
  assert(stage_count >= 1U && stage_count <= TmaMaximumBenchmarkStages);
  const std::size_t dynamic_shared_bytes = stage_count * (transfer_bytes + sizeof(std::uint64_t));
  assert(dynamic_shared_bytes <= properties.sharedMemPerBlockOptin);

  switch (stage_count) {
    case 1U:
      return run_bandwidth_tma_bulk_read_impl<1U>(options, properties, resources, buffer_bytes, tile_count,
                                                  dynamic_shared_bytes);
    case 2U:
      return run_bandwidth_tma_bulk_read_impl<2U>(options, properties, resources, buffer_bytes, tile_count,
                                                  dynamic_shared_bytes);
    case 3U:
      return run_bandwidth_tma_bulk_read_impl<3U>(options, properties, resources, buffer_bytes, tile_count,
                                                  dynamic_shared_bytes);
    case 4U:
      return run_bandwidth_tma_bulk_read_impl<4U>(options, properties, resources, buffer_bytes, tile_count,
                                                  dynamic_shared_bytes);
    default:
      assert(false);
      return false;
  }
}

}  // namespace scinthil::microbenchmark
