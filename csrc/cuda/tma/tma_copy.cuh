#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/tensor.hpp>

namespace scinthil::tma {

using namespace cute;

template <class ElementType, class SmemLayout>
struct TmaCopySharedStorage {
  cute::ArrayEngine<ElementType, cute::cosize_v<SmemLayout>> smem;
  alignas(16) cute::uint64_t tma_load_mbar[1];
};

template <int ThreadsPerBlock, class Element, class SmemLayout>
__launch_bounds__(ThreadsPerBlock) __global__
    void tma_copy_kernel(CUTE_GRID_CONSTANT CUtensorMap const input_map,
                         CUTE_GRID_CONSTANT CUtensorMap const output_map, SmemLayout smem_layout) {
  extern __shared__ char shared_memory[];
  using Storage = TmaCopySharedStorage<Element, SmemLayout>;
  Storage& storage = *reinterpret_cast<Storage*>(shared_memory);

  Tensor sA = make_tensor(make_smem_ptr(storage.smem.begin()), smem_layout);
  uint64_t* tma_load_mbar = storage.tma_load_mbar;

  constexpr int tile_elements = cute::cosize_v<SmemLayout>;
  const int stage = static_cast<int>(blockIdx.x);
  const int tma_coord = stage * tile_elements;

  if (threadIdx.x == 0) {
    constexpr int tma_transaction_bytes = static_cast<int>(sizeof(Element) * tile_elements);
    tma_load_mbar[0] = 0;
    initialize_barrier(tma_load_mbar[0], 1);
    set_barrier_transaction_bytes(tma_load_mbar[0], tma_transaction_bytes);
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    SM90_TMA_LOAD::copy(&input_map, tma_load_mbar, static_cast<uint64_t>(TMA::CacheHintSm90::EVICT_NORMAL),
                        raw_pointer_cast(sA.data()), tma_coord);
  }

  wait_barrier(tma_load_mbar[0], 0);

  if (threadIdx.x == 0) {
    SM90_TMA_STORE::copy(&output_map, raw_pointer_cast(sA.data()), tma_coord);
    tma_store_arrive();
  }
  tma_store_wait<0>();
}

}  // namespace scinthil::tma
