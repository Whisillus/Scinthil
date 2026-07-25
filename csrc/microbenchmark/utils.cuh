#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>

#include "utils/utils.cuh"

namespace scinthil::microbenchmark {

inline constexpr std::size_t Byte2KByte{1024U};
inline constexpr std::size_t Byte2MByte{1024U * 1024U};
inline constexpr std::size_t SharedMemoryMinimumSizeKib{4U};

struct RunMicrobenchmarkOptions {
  std::size_t size_mib{256};
  std::size_t size_kib{16};
  int sweeps{4096};
  int warmup{5};
  int repeats{50};
  int device{0};
};

template <typename Resources, typename Launch>
[[nodiscard]] bool measure(const RunMicrobenchmarkOptions& options, Resources* resources, Launch launch,
                           float* elapsed_ms) {
  std::size_t launch_index{0};
  for (int iteration{0}; iteration < options.warmup; ++iteration) {
    if (!launch(launch_index++)) {
      return false;
    }
  }
  if (!SCINTHIL_CUDA_CHECK(cudaStreamSynchronize(resources->stream))) {
    return false;
  }
  if (!SCINTHIL_CUDA_CHECK(cudaEventRecord(resources->start, resources->stream))) {
    return false;
  }
  for (int iteration{0}; iteration < options.repeats; ++iteration) {
    if (!launch(launch_index++)) {
      return false;
    }
  }
  if (!SCINTHIL_CUDA_CHECK(cudaEventRecord(resources->stop, resources->stream))) {
    return false;
  }
  if (!SCINTHIL_CUDA_CHECK(cudaEventSynchronize(resources->stop))) {
    return false;
  }
  if (!SCINTHIL_CUDA_CHECK(cudaEventElapsedTime(elapsed_ms, resources->start, resources->stop))) {
    return false;
  }
  if (*elapsed_ms <= 0.0F) {
    std::fprintf(stderr, "CUDA event timing returned a non-positive duration\n");
    return false;
  }
  return true;
}

}  // namespace scinthil::microbenchmark
