#pragma once

#include <cuda_runtime.h>
#include <tvm/ffi/tvm_ffi.h>

inline void checkCUDA(cudaError_t result, const char* operation) {
    TVM_FFI_CHECK(result == cudaSuccess)
        << operation << " failed: " << cudaGetErrorString(result);
}
