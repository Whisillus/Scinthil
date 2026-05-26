#pragma once

#include <cuda_runtime.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>
#include <limits>

namespace scinthil::cuda::tvm_utils {

inline bool isFloat32(DLDataType dtype) {
    return dtype.code == kDLFloat && dtype.bits == 32 && dtype.lanes == 1;
}

inline bool hasSameShape(const tvm::ffi::TensorView& lhs,
                         const tvm::ffi::TensorView& rhs) {
    if (lhs.ndim() != rhs.ndim()) {
        return false;
    }
    for (int i = 0; i < lhs.ndim(); ++i) {
        if (lhs.size(i) != rhs.size(i)) {
            return false;
        }
    }
    return true;
}

inline void validateCUDAFloat32Tensor(const tvm::ffi::TensorView& tensor,
                                      const char* name) {
    TVM_FFI_CHECK(tensor.data_ptr() != nullptr) << name << " must have data";
    TVM_FFI_CHECK(tensor.device().device_type == kDLCUDA)
        << name << " must be a CUDA tensor";
    TVM_FFI_CHECK(isFloat32(tensor.dtype())) << name << " must be float32";
    TVM_FFI_CHECK(tensor.IsContiguous()) << name << " must be contiguous";
}

inline void validateSameDevice(const tvm::ffi::TensorView& lhs,
                               const char* lhs_name,
                               const tvm::ffi::TensorView& rhs,
                               const char* rhs_name) {
    TVM_FFI_CHECK(lhs.device().device_type == rhs.device().device_type &&
                  lhs.device().device_id == rhs.device().device_id)
        << lhs_name << " and " << rhs_name << " must be on the same device";
}

inline void validateZeroByteOffset(const tvm::ffi::TensorView& tensor,
                                   const char* name) {
    TVM_FFI_CHECK(tensor.byte_offset() == 0)
        << name << " must have zero byte_offset";
}

inline void validateElementCountWithinInt(const tvm::ffi::TensorView& tensor,
                                          const char* operation) {
    TVM_FFI_CHECK(tensor.numel() <=
                  static_cast<int64_t>(std::numeric_limits<int>::max()))
        << operation << " supports at most INT_MAX elements";
}

inline cudaStream_t currentCUDAStream(DLDevice device) {
    void* stream_ptr = TVMFFIEnvGetStream(device.device_type, device.device_id);
    return reinterpret_cast<cudaStream_t>(stream_ptr);
}

}  // namespace scinthil::cuda::tvm_utils
