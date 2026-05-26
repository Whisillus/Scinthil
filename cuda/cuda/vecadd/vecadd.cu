#include "../tvm_utils.cuh"
#include "../utils.cuh"

#include <cuda_runtime.h>
#include <tvm/ffi/tvm_ffi.h>

#include <cstdint>

namespace {

constexpr int kThreadsPerBlock = 256;
namespace tvm_utils = scinthil::cuda::tvm_utils;

__global__ void vecaddKernel(const float* a, const float* b, float* c, int n) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

void validateVecaddTensors(const tvm::ffi::TensorView& output,
                           const tvm::ffi::TensorView& a,
                           const tvm::ffi::TensorView& b) {
    tvm_utils::validateCUDAFloat32Tensor(output, "output");
    tvm_utils::validateCUDAFloat32Tensor(a, "a");
    tvm_utils::validateCUDAFloat32Tensor(b, "b");

    tvm_utils::validateSameDevice(output, "output", a, "a");
    tvm_utils::validateSameDevice(a, "a", b, "b");
    TVM_FFI_CHECK(tvm_utils::hasSameShape(output, a))
        << "output must have the same shape as a and b";
    TVM_FFI_CHECK(tvm_utils::hasSameShape(a, b))
        << "a and b must have the same shape";
    tvm_utils::validateZeroByteOffset(output, "output");
    tvm_utils::validateZeroByteOffset(a, "a");
    tvm_utils::validateZeroByteOffset(b, "b");
    tvm_utils::validateElementCountWithinInt(a, "vecadd");
}

}  // namespace

void vecadd(tvm::ffi::TensorView output,
            tvm::ffi::TensorView a,
            tvm::ffi::TensorView b) {
    validateVecaddTensors(output, a, b);

    const int64_t count = a.numel();
    tvm::ffi::CUDADeviceGuard device_guard(a.device().device_id);

    if (count == 0) {
        return;
    }

    cudaStream_t stream = tvm_utils::currentCUDAStream(a.device());
    const int block_count =
        (static_cast<int>(count) + kThreadsPerBlock - 1) / kThreadsPerBlock;

    vecaddKernel<<<block_count, kThreadsPerBlock, 0, stream>>>(
        static_cast<const float*>(a.data_ptr()),
        static_cast<const float*>(b.data_ptr()),
        static_cast<float*>(output.data_ptr()), static_cast<int>(count));
    checkCUDA(cudaGetLastError(), "vecaddKernel launch");
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(vecadd, vecadd);
