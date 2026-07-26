#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "bandwidth_utils.cuh"
#include "ptx.cuh"

namespace scinthil::microbenchmark {

__global__ void bandwidth_tma_tensor_read_write_kernel(const __grid_constant__ CUtensorMap input_map,
                                                       const __grid_constant__ CUtensorMap output_map,
                                                       std::size_t tile_count, unsigned int transfer_bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  extern __shared__ __align__(128) unsigned char shared_storage[];
  auto* read_buffer = shared_storage;
  auto* write_buffer = shared_storage + transfer_bytes;
  auto* barrier = reinterpret_cast<std::uint64_t*>(shared_storage + 2U * transfer_bytes);
  if (ptx::elect_one_sync()) {
    ptx::mbarrier_init(barrier);
    ptx::fence_mbarrier_init();
  }
  __syncwarp();

  unsigned int phase{0};
  const int coordinate_rows = static_cast<int>(transfer_bytes / 256U);
  for (std::size_t tile_index = blockIdx.x; tile_index < tile_count; tile_index += gridDim.x) {
    const int coordinate1 = static_cast<int>(tile_index) * coordinate_rows;
    if (ptx::elect_one_sync()) {
      ptx::mbarrier_arrive_expect_tx(barrier, transfer_bytes);
      ptx::tma_tensor_global_to_shared(&input_map, 0, coordinate1, read_buffer, barrier);
      ptx::tma_tensor_shared_to_global(&output_map, 0, coordinate1, write_buffer);
      ptx::tma_store_commit();
    }
    ptx::mbarrier_wait(barrier, phase);
    ptx::tma_store_wait();
    phase ^= 1U;
  }
#else
  static_cast<void>(input_map);
  static_cast<void>(output_map);
  static_cast<void>(tile_count);
  static_cast<void>(transfer_bytes);
#endif
}

[[nodiscard]] inline bool run_bandwidth_tma_tensor_read_write(const RunMicrobenchmarkOptions& options,
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
  const std::size_t dynamic_shared_bytes = 2U * transfer_bytes + sizeof(std::uint64_t);
  if (dynamic_shared_bytes > properties.sharedMemPerBlockOptin) {
    std::fprintf(stderr, "TMA shared-memory allocation exceeds the device per-block limit\n");
    return false;
  }
  if (!SCINTHIL_CUDA_CHECK(cudaFuncSetAttribute(bandwidth_tma_tensor_read_write_kernel,
                                                cudaFuncAttributePreferredSharedMemoryCarveout,
                                                cudaSharedmemCarveoutMaxShared))) {
    return false;
  }
  if (dynamic_shared_bytes > properties.sharedMemPerBlock &&
      !SCINTHIL_CUDA_CHECK(cudaFuncSetAttribute(bandwidth_tma_tensor_read_write_kernel,
                                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                static_cast<int>(dynamic_shared_bytes)))) {
    return false;
  }

  if (!SCINTHIL_CUDA_DRIVER_CHECK(cuInit(0))) {
    return false;
  }
  const std::uint64_t global_dimensions[]{256U, buffer_bytes / 256U};
  const std::uint64_t global_strides[]{256U};
  const std::uint32_t box_dimensions[]{256U, static_cast<std::uint32_t>(transfer_bytes / 256U)};
  const std::uint32_t element_strides[]{1U, 1U};
  std::vector<CUtensorMap> input_maps(resources.workspace_count);
  std::vector<CUtensorMap> output_maps(resources.workspace_count);
  auto* input = static_cast<unsigned char*>(resources.input);
  auto* output = static_cast<unsigned char*>(resources.output);
  for (std::size_t workspace_index{0}; workspace_index < resources.workspace_count; ++workspace_index) {
    if (!SCINTHIL_CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(
            &input_maps[workspace_index], CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, input + workspace_index * buffer_bytes,
            global_dimensions, global_strides, box_dimensions, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)) ||
        !SCINTHIL_CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(
            &output_maps[workspace_index], CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, output + workspace_index * buffer_bytes,
            global_dimensions, global_strides, box_dimensions, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE))) {
      return false;
    }
  }

  const dim3 blocks{static_cast<unsigned int>(properties.multiProcessorCount), 1U, 1U};
  const dim3 threads{32U, 1U, 1U};
  float elapsed_ms{0.0F};
  if (!measure(
          options, resources,
          [=, &resources, &input_maps, &output_maps](std::size_t launch_index) {
            const std::size_t workspace_index = launch_index % resources.workspace_count;
            bandwidth_tma_tensor_read_write_kernel<<<blocks, threads, dynamic_shared_bytes, resources.stream>>>(
                input_maps[workspace_index], output_maps[workspace_index], tile_count,
                static_cast<unsigned int>(transfer_bytes));
            return SCINTHIL_CUDA_CHECK(cudaGetLastError());
          },
          elapsed_ms)) {
    return false;
  }

  print_tma_bandwidth_benchmark_result("TMA tensor read+write", options, properties, resources.workspace_count,
                                       elapsed_ms, 2.0);
  return true;
}

}  // namespace scinthil::microbenchmark
