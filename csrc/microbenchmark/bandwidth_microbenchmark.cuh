#pragma once

#include <cuda_runtime.h>

#include <cstdio>

#include "args.cuh"
#include "bandwidth_global_memory_read.cuh"
#include "bandwidth_global_memory_read_write.cuh"
#include "bandwidth_global_memory_write.cuh"
#include "bandwidth_shared_memory_read.cuh"
#include "bandwidth_shared_memory_read_write.cuh"
#include "bandwidth_shared_memory_write.cuh"
#include "bandwidth_utils.cuh"

namespace scinthil::microbenchmark {

struct BandwidthResourceRequirements {
  bool needs_input{false};
  bool needs_output{false};
};

[[nodiscard]] inline BandwidthResourceRequirements get_bandwidth_resource_requirements(Benchmark benchmark) {
  switch (benchmark) {
    case Benchmark::GlobalMemoryRead:
      return {.needs_input = true, .needs_output = false};
    case Benchmark::GlobalMemoryWrite:
      return {.needs_input = false, .needs_output = true};
    case Benchmark::GlobalMemoryReadWrite:
      return {.needs_input = true, .needs_output = true};
    case Benchmark::SharedMemoryRead:
    case Benchmark::SharedMemoryWrite:
    case Benchmark::SharedMemoryReadWrite:
      return {};
  }
  return {};
}

[[nodiscard]] inline bool initialize_bandwidth_resources(const Arguments& arguments, BandwidthResources* resources,
                                                         cudaDeviceProp* properties) {
  const BandwidthResourceRequirements requirements = get_bandwidth_resource_requirements(arguments.benchmark);
  return resources->initialize_resources(arguments.options, requirements.needs_input, requirements.needs_output,
                                         properties);
}

[[nodiscard]] inline bool run_selected_bandwidth_benchmark(const Arguments& arguments, const cudaDeviceProp& properties,
                                                           BandwidthResources* resources) {
  switch (arguments.benchmark) {
    case Benchmark::GlobalMemoryRead:
      return run_bandwidth_global_memory_read(arguments.options, properties, resources);
    case Benchmark::GlobalMemoryWrite:
      return run_bandwidth_global_memory_write(arguments.options, properties, resources);
    case Benchmark::GlobalMemoryReadWrite:
      return run_bandwidth_global_memory_read_write(arguments.options, properties, resources);
    case Benchmark::SharedMemoryRead:
      return run_bandwidth_shared_memory_read(arguments.options, properties, resources);
    case Benchmark::SharedMemoryWrite:
      return run_bandwidth_shared_memory_write(arguments.options, properties, resources);
    case Benchmark::SharedMemoryReadWrite:
      return run_bandwidth_shared_memory_read_write(arguments.options, properties, resources);
  }
  std::fprintf(stderr, "invalid bandwidth benchmark selection\n");
  return false;
}

[[nodiscard]] inline bool run_bandwidth_microbenchmark(const Arguments& arguments) {
  BandwidthResources resources{};
  cudaDeviceProp properties{};
  const bool initialize_success = initialize_bandwidth_resources(arguments, &resources, &properties);
  const bool run_success = initialize_success && run_selected_bandwidth_benchmark(arguments, properties, &resources);
  const bool cleanup_success = resources.release_resources();
  return run_success && cleanup_success;
}

}  // namespace scinthil::microbenchmark
