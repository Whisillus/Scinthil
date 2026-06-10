#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstdio>
#include <cstdlib>
#include <cute/tensor.hpp>
#include <type_traits>

#include "cuda/tma/tma_copy.cuh"
#include "utils/utils.cuh"

namespace scinthil::example::tma {

using namespace cute;

template <class Element>
[[nodiscard]] constexpr CUtensorMapDataType tma_data_type() {
  if constexpr (std::is_same_v<Element, float>) {
    return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else {
    static_assert(std::is_same_v<Element, float>, "Unsupported TMA element type");
  }
}

template <class Element>
[[nodiscard]] int tma_copy() {
  constexpr int kTileElements{256};
  constexpr int kNumTiles{1024};
  constexpr int kThreadsPerBlock{128};
  constexpr int kTotalElements{kTileElements * kNumTiles};

  thrust::host_vector<Element> host_input(kTotalElements);
  for (int i = 0; i < kTotalElements; ++i) {
    host_input[i] = static_cast<Element>(i);
  }

  thrust::device_vector<Element> device_input = host_input;
  thrust::device_vector<Element> device_output(kTotalElements);
  Element* device_input_ptr = thrust::raw_pointer_cast(device_input.data());
  Element* device_output_ptr = thrust::raw_pointer_cast(device_output.data());

  auto smem_layout = make_layout(Int<kTileElements>{});

  const uint64_t global_dim[] = {kTotalElements};
  const uint64_t* global_strides = nullptr;
  const uint32_t box_dim[] = {kTileElements};
  const uint32_t element_strides[] = {1};

  alignas(64) CUtensorMap input_map{};
  alignas(64) CUtensorMap output_map{};
  bool ok{SCINTHIL_CUDA_DRIVER_CHECK(
      cuTensorMapEncodeTiled(&input_map, tma_data_type<Element>(), 1, device_input_ptr, global_dim, global_strides,
                             box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE))};
  ok = ok && SCINTHIL_CUDA_DRIVER_CHECK(cuTensorMapEncodeTiled(
                 &output_map, tma_data_type<Element>(), 1, device_output_ptr, global_dim, global_strides, box_dim,
                 element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                 CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  constexpr int smem_size =
      static_cast<int>(sizeof(scinthil::tma::TmaCopySharedStorage<Element, decltype(smem_layout)>));
  scinthil::tma::tma_copy_kernel<kThreadsPerBlock, Element>
      <<<kNumTiles, kThreadsPerBlock, smem_size>>>(input_map, output_map, smem_layout);
  ok = ok && SCINTHIL_CUDA_CHECK(cudaGetLastError());
  ok = ok && SCINTHIL_CUDA_CHECK(cudaDeviceSynchronize());

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}

}  // namespace scinthil::example::tma

int main() { return scinthil::example::tma::tma_copy<float>(); }
