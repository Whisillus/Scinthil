#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>

namespace scinthil {

[[nodiscard]] inline bool check_cuda(cudaError_t error, const char* expression, const char* file, int line) {
  if (error == cudaSuccess) {
    return true;
  }

  std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line, expression, cudaGetErrorString(error));
  return false;
}

[[nodiscard]] inline bool check_cuda_driver(CUresult result, const char* expression, const char* file, int line) {
  if (result == CUDA_SUCCESS) {
    return true;
  }

  const char* error_name{nullptr};
  const char* error_string{nullptr};
  cuGetErrorName(result, &error_name);
  cuGetErrorString(result, &error_string);
  std::fprintf(stderr, "CUDA driver error at %s:%d for %s: %s (%s)\n", file, line, expression,
               error_name != nullptr ? error_name : "unknown", error_string != nullptr ? error_string : "unknown");
  return false;
}

}  // namespace scinthil

#ifndef SCINTHIL_CUDA_CHECK
#define SCINTHIL_CUDA_CHECK(expression) (::scinthil::check_cuda((expression), #expression, __FILE__, __LINE__))
#endif

#ifndef SCINTHIL_CUDA_DRIVER_CHECK
#define SCINTHIL_CUDA_DRIVER_CHECK(expression) \
  (::scinthil::check_cuda_driver((expression), #expression, __FILE__, __LINE__))
#endif
