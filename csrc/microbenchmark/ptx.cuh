#pragma once

#include <cuda_runtime.h>

namespace scinthil::microbenchmark::ptx {

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
