#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>

#include "utils.cuh"

namespace scinthil::microbenchmark {

inline void print_bandwidth_benchmark_result(const char* name, const RunMicrobenchmarkOptions& options,
                                             const cudaDeviceProp& properties, float elapsed_ms,
                                             double traffic_multiplier) {
  const double total_bytes = traffic_multiplier * static_cast<double>(options.size_mib * Byte2MByte) * options.repeats;
  const double bandwidth_gb_s = total_bytes / (static_cast<double>(elapsed_ms) / 1000.0) / 1.0e9;
  std::printf("%s: device=%d (%s), size=%zu MiB, traffic=%.0fx, repeats=%d, average=%.3f ms, %.2f GB/s\n", name,
              options.device, properties.name, options.size_mib, traffic_multiplier, options.repeats,
              elapsed_ms / options.repeats, bandwidth_gb_s);
}

struct BandwidthResources {
  void* input{nullptr};
  void* output{nullptr};
  cudaEvent_t start{nullptr};
  cudaEvent_t stop{nullptr};
  cudaStream_t stream{nullptr};

  [[nodiscard]] bool initialize_resources(const RunMicrobenchmarkOptions& options, bool needs_input, bool needs_output,
                                          cudaDeviceProp* properties) {
    const std::size_t bytes = options.size_mib * Byte2MByte;
    if (!SCINTHIL_CUDA_CHECK(cudaSetDevice(options.device))) {
      return false;
    }
    if (!SCINTHIL_CUDA_CHECK(cudaGetDeviceProperties(properties, options.device))) {
      return false;
    }
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
      if (!SCINTHIL_CUDA_CHECK(cudaMalloc(&input, bytes))) {
        return false;
      }
      if (!SCINTHIL_CUDA_CHECK(cudaMemsetAsync(input, 1, bytes, stream))) {
        return false;
      }
    }
    if (needs_output && !SCINTHIL_CUDA_CHECK(cudaMalloc(&output, bytes))) {
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

}  // namespace scinthil::microbenchmark
