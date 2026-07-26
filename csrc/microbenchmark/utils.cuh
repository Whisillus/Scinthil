#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

#include "utils/utils.cuh"

namespace scinthil::microbenchmark {

inline constexpr std::size_t Byte2KByte{1024U};
inline constexpr std::size_t Byte2MByte{1024U * 1024U};
inline constexpr std::size_t L1CacheMinimumSizeKib{4U};
inline constexpr std::size_t SharedMemorySizeGranularityKib{16U};
inline constexpr std::size_t TmaMinimumTransferKib{1U};
inline constexpr std::size_t TmaMaximumTransferKib{16U};
inline constexpr unsigned int TmaMaximumBenchmarkStages{4U};
inline constexpr unsigned int GlobalMemoryAccessesPerThreadIteration{4U};
inline constexpr int SharedMemoryPassesPerLaunchGranularity{4};

enum class TmaKind {
  None,
  Bulk,
  Tensor,
};

struct RunMicrobenchmarkOptions {
  std::size_t global_mib_per_buffer{256};
  std::size_t l1_kib_per_sm{48};
  std::size_t l2_working_set_mib{0};
  std::size_t shared_kib_per_block{16};
  TmaKind tma_kind{TmaKind::None};
  unsigned int tma_benchmark_stages{TmaMaximumBenchmarkStages};
  int passes_per_launch{4096};
  int warmup{5};
  int repeats{50};
  int device{0};
};

struct BenchmarkResult {
  std::size_t num_iter{0};
  std::uint64_t single_iter_read_byte{0};
  std::uint64_t single_iter_write_byte{0};
  std::uint64_t total_read_byte{0};
  std::uint64_t total_write_byte{0};
  std::vector<float> iter_latency_us{};
  double total_latency_us{0.0};
  double mean_latency_us{0.0};
  double median_latency_us{0.0};
  double minimum_latency_us{0.0};
  double maximum_latency_us{0.0};
  double standard_deviation_latency_us{0.0};

  void finalize() {
    num_iter = iter_latency_us.size();
    total_read_byte = single_iter_read_byte * num_iter;
    total_write_byte = single_iter_write_byte * num_iter;
    total_latency_us = std::accumulate(iter_latency_us.begin(), iter_latency_us.end(), 0.0);
    mean_latency_us = total_latency_us / static_cast<double>(num_iter);

    std::vector<float> sorted_latency_us = iter_latency_us;
    std::sort(sorted_latency_us.begin(), sorted_latency_us.end());
    minimum_latency_us = sorted_latency_us.front();
    maximum_latency_us = sorted_latency_us.back();
    const std::size_t middle = num_iter / 2U;
    median_latency_us = num_iter % 2U == 0U
                            ? (static_cast<double>(sorted_latency_us[middle - 1U]) + sorted_latency_us[middle]) / 2.0
                            : sorted_latency_us[middle];

    double squared_deviation_sum{0.0};
    for (const float latency_us : iter_latency_us) {
      const double deviation = latency_us - mean_latency_us;
      squared_deviation_sum += deviation * deviation;
    }
    standard_deviation_latency_us = std::sqrt(squared_deviation_sum / static_cast<double>(num_iter));
  }
};

template <typename Resources, typename Launch>
[[nodiscard]] bool measure(const RunMicrobenchmarkOptions& options, Resources& resources, Launch launch,
                           std::uint64_t single_iter_read_byte, std::uint64_t single_iter_write_byte,
                           BenchmarkResult& result) {
  result = {};
  result.single_iter_read_byte = single_iter_read_byte;
  result.single_iter_write_byte = single_iter_write_byte;

  std::size_t launch_index{0};
  for (int iteration{0}; iteration < options.warmup; ++iteration) {
    if (!launch(launch_index++)) {
      return false;
    }
  }
  if (!SCINTHIL_CUDA_CHECK(cudaStreamSynchronize(resources.stream))) {
    return false;
  }

  std::vector<cudaEvent_t> events(static_cast<std::size_t>(options.repeats) + 1U, nullptr);
  const auto release_events = [&events]() {
    bool success{true};
    for (const cudaEvent_t event : events) {
      if (event != nullptr) {
        success = SCINTHIL_CUDA_CHECK(cudaEventDestroy(event)) && success;
      }
    }
    return success;
  };
  for (cudaEvent_t& event : events) {
    if (!SCINTHIL_CUDA_CHECK(cudaEventCreate(&event))) {
      release_events();
      return false;
    }
  }

  if (!SCINTHIL_CUDA_CHECK(cudaEventRecord(events.front(), resources.stream))) {
    release_events();
    return false;
  }
  for (int iteration{0}; iteration < options.repeats; ++iteration) {
    if (!launch(launch_index++)) {
      release_events();
      return false;
    }
    if (!SCINTHIL_CUDA_CHECK(cudaEventRecord(events[static_cast<std::size_t>(iteration) + 1U], resources.stream))) {
      release_events();
      return false;
    }
  }
  if (!SCINTHIL_CUDA_CHECK(cudaEventSynchronize(events.back()))) {
    release_events();
    return false;
  }

  result.iter_latency_us.reserve(static_cast<std::size_t>(options.repeats));
  for (std::size_t iteration{0}; iteration < static_cast<std::size_t>(options.repeats); ++iteration) {
    float latency_ms{0.0F};
    if (!SCINTHIL_CUDA_CHECK(cudaEventElapsedTime(&latency_ms, events[iteration], events[iteration + 1U]))) {
      release_events();
      return false;
    }
    const float latency_us = latency_ms * 1000.0F;
    if (latency_us <= 0.0F) {
      std::fprintf(stderr, "CUDA event timing returned a non-positive duration\n");
      release_events();
      return false;
    }
    result.iter_latency_us.push_back(latency_us);
  }
  result.finalize();
  return release_events();
}

}  // namespace scinthil::microbenchmark
