#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <cstddef>
#include <cstdio>

#include "utils.cuh"

namespace scinthil::microbenchmark {

[[nodiscard]] inline double calculate_bandwidth_gb_s(std::uint64_t bytes, double latency_us) {
  return static_cast<double>(bytes) / latency_us / 1.0e3;
}

inline void print_benchmark_result_statistics(const BenchmarkResult& result) {
  const std::uint64_t single_iter_aggregate_byte = result.single_iter_read_byte + result.single_iter_write_byte;
  std::printf(
      "  iterations=%zu, read/iteration=%llu B, write/iteration=%llu B, total-read=%llu B, total-write=%llu B\n"
      "  latency-us: total=%.3f, mean=%.3f, median=%.3f, min=%.3f, max=%.3f, stddev=%.3f\n"
      "  bandwidth-GB/s: mean(read=%.2f, write=%.2f, aggregate=%.2f), "
      "median(read=%.2f, write=%.2f, aggregate=%.2f), best(read=%.2f, write=%.2f, aggregate=%.2f)\n",
      result.num_iter, static_cast<unsigned long long>(result.single_iter_read_byte),
      static_cast<unsigned long long>(result.single_iter_write_byte),
      static_cast<unsigned long long>(result.total_read_byte), static_cast<unsigned long long>(result.total_write_byte),
      result.total_latency_us, result.mean_latency_us, result.median_latency_us, result.minimum_latency_us,
      result.maximum_latency_us, result.standard_deviation_latency_us,
      calculate_bandwidth_gb_s(result.single_iter_read_byte, result.mean_latency_us),
      calculate_bandwidth_gb_s(result.single_iter_write_byte, result.mean_latency_us),
      calculate_bandwidth_gb_s(single_iter_aggregate_byte, result.mean_latency_us),
      calculate_bandwidth_gb_s(result.single_iter_read_byte, result.median_latency_us),
      calculate_bandwidth_gb_s(result.single_iter_write_byte, result.median_latency_us),
      calculate_bandwidth_gb_s(single_iter_aggregate_byte, result.median_latency_us),
      calculate_bandwidth_gb_s(result.single_iter_read_byte, result.minimum_latency_us),
      calculate_bandwidth_gb_s(result.single_iter_write_byte, result.minimum_latency_us),
      calculate_bandwidth_gb_s(single_iter_aggregate_byte, result.minimum_latency_us));
}

inline void print_bandwidth_benchmark_result(const char* name, const RunMicrobenchmarkOptions& options,
                                             const cudaDeviceProp& properties, std::size_t workspace_count,
                                             unsigned int block_count, unsigned int threads_per_block,
                                             std::size_t index_bits, const BenchmarkResult& result) {
  std::printf(
      "%s: device=%d (%s), buffer=%zu MiB, workspaces=%zu, blocks=%u, threads/block=%u, index=%zu-bit, "
      "accesses/thread-iteration=%u\n",
      name, options.device, properties.name, options.global_mib_per_buffer, workspace_count, block_count,
      threads_per_block, index_bits, GlobalMemoryAccessesPerThreadIteration);
  print_benchmark_result_statistics(result);
}

struct BandwidthResources {
  void* input{nullptr};
  void* output{nullptr};
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

inline void print_tma_bandwidth_benchmark_result(const char* name, const RunMicrobenchmarkOptions& options,
                                                 const cudaDeviceProp& properties, std::size_t workspace_count,
                                                 unsigned int stage_count, const BenchmarkResult& result) {
  std::printf(
      "%s: device=%d (%s), buffer=%zu MiB, workspaces=%zu, transfer=%zu KiB, benchmark-stages=%u, blocks=%u, "
      "threads/block=%u\n",
      name, options.device, properties.name, options.global_mib_per_buffer, workspace_count,
      options.shared_kib_per_block, stage_count, properties.multiProcessorCount, 32U);
  print_benchmark_result_statistics(result);
}

inline void print_shared_memory_bandwidth_benchmark_result(const char* name, const RunMicrobenchmarkOptions& options,
                                                           const cudaDeviceProp& properties, unsigned int block_count,
                                                           const BenchmarkResult& result) {
  std::printf("%s: device=%d (%s), shared=%zu KiB/block, blocks=%u, passes/launch=%d\n", name, options.device,
              properties.name, options.shared_kib_per_block, block_count, options.passes_per_launch);
  print_benchmark_result_statistics(result);
}

inline void print_l1_cache_bandwidth_benchmark_result(const RunMicrobenchmarkOptions& options,
                                                      const cudaDeviceProp& properties, unsigned int blocks_per_sm,
                                                      std::size_t bytes_per_block, const BenchmarkResult& result) {
  const std::size_t bytes_per_sm = bytes_per_block * blocks_per_sm;
  std::printf(
      "L1 cache read-only: device=%d (%s), working-set=%.2f KiB/SM, slice=%.2f KiB/block, blocks/SM=%u, "
      "passes/launch=%d\n",
      options.device, properties.name, static_cast<double>(bytes_per_sm) / Byte2KByte,
      static_cast<double>(bytes_per_block) / Byte2KByte, blocks_per_sm, options.passes_per_launch);
  print_benchmark_result_statistics(result);
}

inline void print_l2_cache_bandwidth_benchmark_result(const RunMicrobenchmarkOptions& options,
                                                      const cudaDeviceProp& properties, std::size_t working_set_bytes,
                                                      unsigned int block_count, const BenchmarkResult& result) {
  std::printf("L2 cache read-only: device=%d (%s), working-set=%.2f MiB, blocks=%u, passes/launch=%d\n", options.device,
              properties.name, static_cast<double>(working_set_bytes) / Byte2MByte, block_count,
              options.passes_per_launch);
  print_benchmark_result_statistics(result);
}

}  // namespace scinthil::microbenchmark
