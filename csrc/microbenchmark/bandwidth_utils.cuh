#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdio>

#include "utils.cuh"

namespace scinthil::microbenchmark {

inline void print_bandwidth_benchmark_result(const char* name, const RunMicrobenchmarkOptions& options,
                                             const cudaDeviceProp& properties, std::size_t workspace_count,
                                             unsigned int block_count, unsigned int threads_per_block,
                                             std::size_t index_bits, float elapsed_ms, double traffic_multiplier) {
  const double total_bytes =
      traffic_multiplier * static_cast<double>(options.global_mib_per_buffer * Byte2MByte) * options.repeats;
  const double bandwidth_gb_s = total_bytes / (static_cast<double>(elapsed_ms) / 1000.0) / 1.0e9;
  std::printf(
      "%s: device=%d (%s), buffer=%zu MiB, workspaces=%zu, blocks=%u, threads/block=%u, index=%zu-bit, "
      "accesses/thread-iteration=%u, traffic=%.0fx, repeats=%d, average=%.3f ms, %.2f GB/s\n",
      name, options.device, properties.name, options.global_mib_per_buffer, workspace_count, block_count,
      threads_per_block, index_bits, GlobalMemoryAccessesPerThreadIteration, traffic_multiplier, options.repeats,
      elapsed_ms / options.repeats, bandwidth_gb_s);
}

struct BandwidthResources {
  void* input{nullptr};
  void* output{nullptr};
  cudaEvent_t start{nullptr};
  cudaEvent_t stop{nullptr};
  cudaStream_t stream{nullptr};
  std::size_t workspace_count{1};

  [[nodiscard]] bool allocate_input(std::size_t bytes) { return SCINTHIL_CUDA_CHECK(cudaMalloc(&input, bytes)); }

  [[nodiscard]] bool allocate_output(std::size_t bytes) { return SCINTHIL_CUDA_CHECK(cudaMalloc(&output, bytes)); }

  [[nodiscard]] bool initialize_resources(const RunMicrobenchmarkOptions& options, bool needs_input, bool needs_output,
                                          cudaDeviceProp& properties) {
    const std::size_t bytes = options.global_mib_per_buffer * Byte2MByte;
    if (!SCINTHIL_CUDA_CHECK(cudaSetDevice(options.device))) {
      return false;
    }
    if (!SCINTHIL_CUDA_CHECK(cudaGetDeviceProperties(&properties, options.device))) {
      return false;
    }

    const std::size_t buffer_count = static_cast<std::size_t>(needs_input) + static_cast<std::size_t>(needs_output);
    if (buffer_count > 0) {
      const std::size_t rotation_bytes = 3U * static_cast<std::size_t>(properties.l2CacheSize);
      if (bytes < rotation_bytes) {
        const std::size_t bytes_per_workspace = bytes * buffer_count;
        if (bytes_per_workspace < rotation_bytes) {
          workspace_count = 1U + rotation_bytes / bytes_per_workspace;
        }
      }
    }
    const std::size_t allocation_bytes = bytes * workspace_count;

    if (!SCINTHIL_CUDA_CHECK(cudaStreamCreate(&stream))) {
      return false;
    }
    if (!SCINTHIL_CUDA_CHECK(cudaEventCreate(&start))) {
      return false;
    }
    if (!SCINTHIL_CUDA_CHECK(cudaEventCreate(&stop))) {
      return false;
    }
    if (needs_input) {
      if (!allocate_input(allocation_bytes)) {
        return false;
      }
      if (!SCINTHIL_CUDA_CHECK(cudaMemsetAsync(input, 1, allocation_bytes, stream))) {
        return false;
      }
    }
    if (needs_output && !allocate_output(allocation_bytes)) {
      return false;
    }
    return true;
  }

  [[nodiscard]] bool release_resources() {
    bool success{true};
    if (stream != nullptr) {
      success = SCINTHIL_CUDA_CHECK(cudaStreamSynchronize(stream)) && success;
    }
    if (input != nullptr) {
      success = SCINTHIL_CUDA_CHECK(cudaFree(input)) && success;
    }
    if (output != nullptr) {
      success = SCINTHIL_CUDA_CHECK(cudaFree(output)) && success;
    }
    if (start != nullptr) {
      success = SCINTHIL_CUDA_CHECK(cudaEventDestroy(start)) && success;
    }
    if (stop != nullptr) {
      success = SCINTHIL_CUDA_CHECK(cudaEventDestroy(stop)) && success;
    }
    if (stream != nullptr) {
      success = SCINTHIL_CUDA_CHECK(cudaStreamDestroy(stream)) && success;
    }
    return success;
  }
};

template <typename Kernel>
[[nodiscard]] bool prepare_global_memory_launch(Kernel kernel, std::size_t element_count,
                                                const cudaDeviceProp& properties, dim3& threads, dim3& blocks) {
  threads = dim3{256U, 1U, 1U};
  int active_blocks_per_multiprocessor{0};
  if (!SCINTHIL_CUDA_CHECK(
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_multiprocessor, kernel, threads.x, 0))) {
    return false;
  }

  const std::size_t elements_per_block_iteration =
      static_cast<std::size_t>(GlobalMemoryAccessesPerThreadIteration) * threads.x;
  assert(element_count % elements_per_block_iteration == 0);
  const std::size_t size_limited_block_count = element_count / elements_per_block_iteration;
  const std::size_t occupancy_block_count =
      static_cast<std::size_t>(active_blocks_per_multiprocessor) * properties.multiProcessorCount;
  std::size_t block_count =
      size_limited_block_count < occupancy_block_count ? size_limited_block_count : occupancy_block_count;
  while (element_count % (elements_per_block_iteration * block_count) != 0) {
    --block_count;
  }
  assert(element_count % (elements_per_block_iteration * block_count) == 0);
  blocks = dim3{static_cast<unsigned int>(block_count), 1U, 1U};
  return true;
}

template <typename Kernel>
[[nodiscard]] bool prepare_shared_memory_launch(Kernel kernel, const RunMicrobenchmarkOptions& options,
                                                const cudaDeviceProp& properties, const dim3& threads, dim3& blocks) {
  const std::size_t shared_memory_bytes = options.shared_kib_per_block * Byte2KByte;
  assert(shared_memory_bytes % (4U * threads.x * sizeof(uint4)) == 0);
  assert(options.passes_per_launch % SharedMemoryPassesPerLaunchGranularity == 0);
  const std::size_t maximum_shared_memory_bytes = properties.sharedMemPerBlockOptin;
  if (shared_memory_bytes > maximum_shared_memory_bytes) {
    std::fprintf(stderr, "requested shared-memory size exceeds the device per-block limit (%zu bytes)\n",
                 maximum_shared_memory_bytes);
    return false;
  }

  if (!SCINTHIL_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                                cudaSharedmemCarveoutMaxShared))) {
    return false;
  }
  if (shared_memory_bytes > properties.sharedMemPerBlock &&
      !SCINTHIL_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                                static_cast<int>(shared_memory_bytes)))) {
    return false;
  }

  int active_blocks_per_multiprocessor{0};
  if (!SCINTHIL_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks_per_multiprocessor, kernel,
                                                                         threads.x, shared_memory_bytes))) {
    return false;
  }

  const std::size_t block_count =
      static_cast<std::size_t>(active_blocks_per_multiprocessor) * properties.multiProcessorCount;
  blocks = dim3{static_cast<unsigned int>(block_count), 1, 1};
  return true;
}

inline void print_shared_memory_bandwidth_benchmark_result(const char* name, const RunMicrobenchmarkOptions& options,
                                                           const cudaDeviceProp& properties, unsigned int block_count,
                                                           float elapsed_ms, double traffic_multiplier) {
  const double total_bytes = traffic_multiplier * static_cast<double>(options.shared_kib_per_block * Byte2KByte) *
                             block_count * options.passes_per_launch * options.repeats;
  const double bandwidth_gb_s = total_bytes / (static_cast<double>(elapsed_ms) / 1000.0) / 1.0e9;
  std::printf(
      "%s: device=%d (%s), shared=%zu KiB/block, blocks=%u, passes/launch=%d, traffic=%.0fx, repeats=%d, "
      "average=%.3f ms, %.2f GB/s\n",
      name, options.device, properties.name, options.shared_kib_per_block, block_count, options.passes_per_launch,
      traffic_multiplier, options.repeats, elapsed_ms / options.repeats, bandwidth_gb_s);
}

inline void print_l1_cache_bandwidth_benchmark_result(const RunMicrobenchmarkOptions& options,
                                                      const cudaDeviceProp& properties, unsigned int blocks_per_sm,
                                                      std::size_t bytes_per_block, float elapsed_ms) {
  const std::size_t bytes_per_sm = bytes_per_block * blocks_per_sm;
  const std::size_t working_set_bytes = bytes_per_sm * static_cast<std::size_t>(properties.multiProcessorCount);
  const double total_bytes = static_cast<double>(working_set_bytes) * options.passes_per_launch * options.repeats;
  const double bandwidth_gb_s = total_bytes / (static_cast<double>(elapsed_ms) / 1000.0) / 1.0e9;
  std::printf(
      "L1 cache read-only: device=%d (%s), working-set=%.2f KiB/SM, slice=%.2f KiB/block, blocks/SM=%u, "
      "passes/launch=%d, repeats=%d, average=%.3f ms, %.2f GB/s\n",
      options.device, properties.name, static_cast<double>(bytes_per_sm) / Byte2KByte,
      static_cast<double>(bytes_per_block) / Byte2KByte, blocks_per_sm, options.passes_per_launch, options.repeats,
      elapsed_ms / options.repeats, bandwidth_gb_s);
}

inline void print_l2_cache_bandwidth_benchmark_result(const RunMicrobenchmarkOptions& options,
                                                      const cudaDeviceProp& properties, std::size_t working_set_bytes,
                                                      unsigned int block_count, float elapsed_ms) {
  const double total_bytes = static_cast<double>(working_set_bytes) * options.passes_per_launch * options.repeats;
  const double bandwidth_gb_s = total_bytes / (static_cast<double>(elapsed_ms) / 1000.0) / 1.0e9;
  std::printf(
      "L2 cache read-only: device=%d (%s), working-set=%.2f MiB, blocks=%u, passes/launch=%d, repeats=%d, "
      "average=%.3f ms, %.2f GB/s\n",
      options.device, properties.name, static_cast<double>(working_set_bytes) / Byte2MByte, block_count,
      options.passes_per_launch, options.repeats, elapsed_ms / options.repeats, bandwidth_gb_s);
}

}  // namespace scinthil::microbenchmark
