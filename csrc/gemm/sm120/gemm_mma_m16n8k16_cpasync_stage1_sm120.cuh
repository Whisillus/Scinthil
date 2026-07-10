#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/tensor.hpp>

#include "gemm/sm120/op_selector.cuh"

namespace scinthil::gemm::sm120 {

using namespace cute;

template <class ElementAB, class SmemLayoutA, class SmemLayoutB>
struct GemmMmaM16N8K16CpAsyncStage1SM120SharedStorage {
  alignas(128) ArrayEngine<ElementAB, cosize_v<SmemLayoutA>> a;
  alignas(128) ArrayEngine<ElementAB, cosize_v<SmemLayoutB>> b;
};

template <class ElementAB, class ElementD, class StrideA, class StrideB, class StrideD>
__launch_bounds__(32, 1) __global__
    void gemm_mma_m16n8k16_cpasync_stage1_sm120(const ElementAB* a, const ElementAB* b, ElementD* d, int m, int n,
                                                int k, StrideA stride_a, StrideB stride_b, StrideD stride_d) {
  using TileShape = Shape<Int<16>, Int<8>, Int<16>>;
  using SmemLayoutA = decltype(make_layout(select<0, 2>(TileShape{}), make_stride(size<2>(TileShape{}), _1{})));
  using SmemLayoutB = decltype(make_layout(select<1, 2>(TileShape{}), make_stride(size<2>(TileShape{}), _1{})));
  using Storage = GemmMmaM16N8K16CpAsyncStage1SM120SharedStorage<ElementAB, SmemLayoutA, SmemLayoutB>;
  using MmaOp = typename MMAOpSelectorRRSM120<ElementAB, ElementAB, ElementD, size<0>(TileShape{}),
                                              size<1>(TileShape{}), size<2>(TileShape{})>::Type;
  constexpr int threads_per_block{32};

  auto problem_shape = make_shape(m, n, k);

  Tensor mA = make_tensor(make_gmem_ptr(a), select<0, 2>(problem_shape), stride_a);
  Tensor mB = make_tensor(make_gmem_ptr(b), select<1, 2>(problem_shape), stride_b);
  Tensor mD = make_tensor(make_gmem_ptr(d), select<0, 1>(problem_shape), stride_d);

  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, TileShape{}, cta_coord, Step<_1, X, _1>{});  // (TileM, TileK, k_tile)
  Tensor gB = local_tile(mB, TileShape{}, cta_coord, Step<X, _1, _1>{});  // (TileN, TileK, k_tile)
  Tensor gD = local_tile(mD, TileShape{}, cta_coord, Step<_1, _1, X>{});  // (TileM, TileN)

  __shared__ Storage storage;
  Tensor sA = make_tensor(make_smem_ptr(storage.a.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(storage.b.begin()), SmemLayoutB{});

  TiledCopy tiled_copy_a = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint128_t>, ElementAB>{},
                                           Layout<Shape<_16, _2>, Stride<_2, _1>>{}, Layout<Shape<_1, _8>>{});
  TiledCopy tiled_copy_b = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<std::uint64_t>, ElementAB>{},
                                           Layout<Shape<_8, _4>, Stride<_4, _1>>{}, Layout<Shape<_1, _4>>{});

  ThrCopy thr_copy_a = tiled_copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tAsA = thr_copy_a.partition_D(sA);

  ThrCopy thr_copy_b = tiled_copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);
  Tensor tBsB = thr_copy_b.partition_D(sB);

  TiledMMA tiled_mma = make_tiled_mma(MmaOp{});
  CUTE_STATIC_ASSERT_V(size(tiled_copy_a) == size(tiled_mma));
  CUTE_STATIC_ASSERT_V(size(tiled_copy_b) == size(tiled_mma));
  CUTE_STATIC_ASSERT_V(size(tiled_mma) == Int<threads_per_block>{});

  ThrMMA thr_mma = tiled_mma.get_slice(threadIdx.x);
  Tensor tDgD = thr_mma.partition_C(gD);
  Tensor tCrA = thr_mma.partition_fragment_A(sA);
  Tensor tCrB = thr_mma.partition_fragment_B(sB);
  Tensor tCrC = thr_mma.make_fragment_C(tDgD);
  clear(tCrC);

  Copy_Atom<SM75_U32x4_LDSM_N, ElementAB> s2r_atom_a;
  Copy_Atom<SM75_U32x2_LDSM_N, ElementAB> s2r_atom_b;
  TiledCopy tiled_s2r_copy_a = make_tiled_copy_A(s2r_atom_a, tiled_mma);
  TiledCopy tiled_s2r_copy_b = make_tiled_copy_B(s2r_atom_b, tiled_mma);

  ThrCopy s2r_thr_copy_a = tiled_s2r_copy_a.get_slice(threadIdx.x);
  Tensor tXsA = s2r_thr_copy_a.partition_S(sA);
  Tensor tXrA = s2r_thr_copy_a.retile_D(tCrA);

  ThrCopy s2r_thr_copy_b = tiled_s2r_copy_b.get_slice(threadIdx.x);
  Tensor tXsB = s2r_thr_copy_b.partition_S(sB);
  Tensor tXrB = s2r_thr_copy_b.retile_D(tCrB);

  const int k_tiles = k / size<2>(TileShape{});
  for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
    copy(tiled_copy_a, tAgA(_, _, _, k_tile), tAsA);
    copy(tiled_copy_b, tBgB(_, _, _, k_tile), tBsB);
    cp_async_fence();
    cp_async_wait<0>();
    __syncwarp();

    copy(s2r_atom_a, tXsA, tXrA);
    copy(s2r_atom_b, tXsB, tXrB);
    cute::gemm(tiled_mma, tCrA, tCrB, tCrC);

    __syncwarp();
  }

  copy(tCrC, tDgD);
}

template <class ElementAB, class ElementD, class StrideA, class StrideB, class StrideD>
[[nodiscard]] cudaError_t launch_gemm_mma_m16n8k16_cpasync_stage1_sm120(const ElementAB* a, const ElementAB* b,
                                                                        ElementD* d, int m, int n, int k,
                                                                        StrideA stride_a, StrideB stride_b,
                                                                        StrideD stride_d,
                                                                        cudaStream_t stream = nullptr) {
  auto tile_shape = make_shape(Int<16>{}, Int<8>{}, Int<16>{});

  if (m <= 0 || n <= 0 || k <= 0) {
    return cudaErrorInvalidValue;
  }
  if (a == nullptr || b == nullptr || d == nullptr) {
    return cudaErrorInvalidValue;
  }
  if ((m % size<0>(tile_shape)) != 0 || (n % size<1>(tile_shape)) != 0 || (k % size<2>(tile_shape)) != 0) {
    return cudaErrorInvalidValue;
  }

  dim3 grid(static_cast<unsigned>(m / size<0>(tile_shape)), static_cast<unsigned>(n / size<1>(tile_shape)));

  gemm_mma_m16n8k16_cpasync_stage1_sm120<ElementAB, ElementD, StrideA, StrideB, StrideD>
      <<<grid, 32, 0, stream>>>(a, b, d, m, n, k, stride_a, stride_b, stride_d);

  return cudaGetLastError();
}

}  // namespace scinthil::gemm::sm120
