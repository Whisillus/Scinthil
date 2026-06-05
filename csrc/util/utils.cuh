#pragma once

#include <cuda_runtime.h>

#include <cstdio>

namespace scinthil {

[[nodiscard]] inline bool check_cuda(cudaError_t error, const char *expression,
                                     const char *file, int line) {
  if (error == cudaSuccess) {
    return true;
  }

  std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line,
               expression, cudaGetErrorString(error));
  return false;
}

} // namespace scinthil

#ifndef SCINTHIL_CUDA_CHECK
#define SCINTHIL_CUDA_CHECK(expression)                                        \
  (::scinthil::check_cuda((expression), #expression, __FILE__, __LINE__))
#endif
