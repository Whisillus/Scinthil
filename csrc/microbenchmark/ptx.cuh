#pragma once

#include <cuda_runtime.h>

namespace scinthil::microbenchmark::ptx {

[[nodiscard]] __device__ __forceinline__ bool elect_one_sync() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  unsigned int elected{0};
  unsigned int lane_id{0};
  asm volatile(
      "{\n\t"
      ".reg .pred is_elected;\n\t"
      "elect.sync %0|is_elected, %2;\n\t"
      "@is_elected mov.u32 %1, 1;\n\t"
      "}"
      : "+r"(lane_id), "+r"(elected)
      : "r"(0xFFFFFFFFU));
  return elected != 0U;
#else
  return (threadIdx.x & 31U) == 0U;
#endif
}

template <typename T>
[[nodiscard]] __device__ __forceinline__ T ldg_128bit(const T* input) {
  static_assert(sizeof(T) == 16);
  static_assert(alignof(T) >= 16);
  T value;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(input)
               : "memory");
  return value;
}

template <typename T>
__device__ __forceinline__ void ldg_128bit_discard_result(const T* input) {
  static_assert(sizeof(T) == 16);
  static_assert(alignof(T) >= 16);
  unsigned int value0;
  unsigned int value1;
  unsigned int value2;
  unsigned int value3;
  asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value0), "=r"(value1), "=r"(value2), "=r"(value3)
               : "l"(input)
               : "memory");
}

template <typename T>
[[nodiscard]] __device__ __forceinline__ T ldg_ca_128bit(const T* input) {
  static_assert(sizeof(T) == 16);
  static_assert(alignof(T) >= 16);
  T value;
  asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(input)
               : "memory");
  return value;
}

template <typename T>
[[nodiscard]] __device__ __forceinline__ T ldg_cg_128bit(const T* input) {
  static_assert(sizeof(T) == 16);
  static_assert(alignof(T) >= 16);
  T value;
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(input)
               : "memory");
  return value;
}

template <typename T>
__device__ __forceinline__ void stg_128bit(T* output, T value) {
  static_assert(sizeof(T) == 16);
  static_assert(alignof(T) >= 16);
  asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(output), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
               : "memory");
}

[[nodiscard]] __device__ __forceinline__ unsigned int cvta_to_shared(const void* pointer) {
  return static_cast<unsigned int>(__cvta_generic_to_shared(pointer));
}

__device__ __forceinline__ void mbarrier_init(uint64_t* barrier) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const unsigned int address = cvta_to_shared(barrier);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" : : "r"(address) : "memory");
#else
  static_cast<void>(barrier);
#endif
}

__device__ __forceinline__ void fence_mbarrier_init() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fence.mbarrier_init.release.cluster;" : : : "memory");
#endif
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* barrier, unsigned int transaction_bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const unsigned int address = cvta_to_shared(barrier);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :
               : "r"(address), "r"(transaction_bytes)
               : "memory");
#else
  static_cast<void>(barrier);
  static_cast<void>(transaction_bytes);
#endif
}

__device__ __forceinline__ void mbarrier_wait(uint64_t* barrier, unsigned int phase) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const unsigned int address = cvta_to_shared(barrier);
  unsigned int complete{0};
  do {
    asm volatile(
        "{\n\t"
        ".reg .pred ready;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 ready, [%1], %2;\n\t"
        "selp.b32 %0, 1, 0, ready;\n\t"
        "}"
        : "=r"(complete)
        : "r"(address), "r"(phase)
        : "memory");
  } while (complete == 0U);
#else
  static_cast<void>(barrier);
  static_cast<void>(phase);
#endif
}

__device__ __forceinline__ void tma_bulk_global_to_shared(const void* global, void* shared, uint64_t* barrier,
                                                          unsigned int transaction_bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const unsigned int shared_address = cvta_to_shared(shared);
  const unsigned int barrier_address = cvta_to_shared(barrier);
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :
               : "r"(shared_address), "l"(global), "r"(transaction_bytes), "r"(barrier_address)
               : "memory");
#else
  static_cast<void>(global);
  static_cast<void>(shared);
  static_cast<void>(barrier);
  static_cast<void>(transaction_bytes);
#endif
}

__device__ __forceinline__ void tma_bulk_shared_to_global(void* global, const void* shared,
                                                          unsigned int transaction_bytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const unsigned int shared_address = cvta_to_shared(shared);
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
               :
               : "l"(global), "r"(shared_address), "r"(transaction_bytes)
               : "memory");
#else
  static_cast<void>(global);
  static_cast<void>(shared);
  static_cast<void>(transaction_bytes);
#endif
}

__device__ __forceinline__ void tma_tensor_global_to_shared(const void* descriptor, int coordinate0, int coordinate1,
                                                            void* shared, uint64_t* barrier) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
  const unsigned int shared_address = cvta_to_shared(shared);
  const unsigned int barrier_address = cvta_to_shared(barrier);
  constexpr uint64_t cache_hint{0x1000000000000000ULL};
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1, {%3, %4}], [%2], %5;"
      :
      : "r"(shared_address), "l"(descriptor), "r"(barrier_address), "r"(coordinate0), "r"(coordinate1), "l"(cache_hint)
      : "memory");
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const unsigned int shared_address = cvta_to_shared(shared);
  const unsigned int barrier_address = cvta_to_shared(barrier);
  constexpr uint64_t cache_hint{0x1000000000000000ULL};
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1, {%3, %4}], [%2], %5;"
      :
      : "r"(shared_address), "l"(descriptor), "r"(barrier_address), "r"(coordinate0), "r"(coordinate1), "l"(cache_hint)
      : "memory");
#else
  static_cast<void>(descriptor);
  static_cast<void>(coordinate0);
  static_cast<void>(coordinate1);
  static_cast<void>(shared);
  static_cast<void>(barrier);
#endif
}

__device__ __forceinline__ void tma_tensor_shared_to_global(const void* descriptor, int coordinate0, int coordinate1,
                                                            const void* shared) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const unsigned int shared_address = cvta_to_shared(shared);
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
               :
               : "l"(descriptor), "r"(shared_address), "r"(coordinate0), "r"(coordinate1)
               : "memory");
#else
  static_cast<void>(descriptor);
  static_cast<void>(coordinate0);
  static_cast<void>(coordinate1);
  static_cast<void>(shared);
#endif
}

__device__ __forceinline__ void tma_store_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.commit_group;" : : : "memory");
#endif
}

template <unsigned int PendingGroups>
__device__ __forceinline__ void tma_store_wait_read() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(PendingGroups) : "memory");
#endif
}

__device__ __forceinline__ void tma_store_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.wait_group 0;" : : : "memory");
#endif
}

template <typename T>
[[nodiscard]] __device__ __forceinline__ T lds_128bit(unsigned int address) {
  static_assert(sizeof(T) == 16);
  static_assert(alignof(T) >= 16);
  T value;
  asm volatile("ld.volatile.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "r"(address)
               : "memory");
  return value;
}

template <typename T>
__device__ __forceinline__ void lds_128bit_discard_result(unsigned int address) {
  static_assert(sizeof(T) == 16);
  static_assert(alignof(T) >= 16);
  unsigned int value0;
  unsigned int value1;
  unsigned int value2;
  unsigned int value3;
  asm volatile("ld.volatile.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value0), "=r"(value1), "=r"(value2), "=r"(value3)
               : "r"(address)
               : "memory");
}

template <typename T>
__device__ __forceinline__ void sts_128bit(unsigned int address, T value) {
  static_assert(sizeof(T) == 16);
  static_assert(alignof(T) >= 16);
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "r"(address), "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
               : "memory");
}

}  // namespace scinthil::microbenchmark::ptx
