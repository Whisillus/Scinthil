#pragma once

#include <cute/arch/mma_sm80.hpp>
#include <cute/numeric/numeric_types.hpp>

namespace scinthil::gemm::sm120 {

template <class ElementA, class ElementB, class ElementC, int MMA_M, int MMA_N, int MMA_K>
struct MMAOpSelectorRRSM120;

template <>
struct MMAOpSelectorRRSM120<cute::half_t, cute::half_t, cute::half_t, 16, 8, 8> {
  using Type = cute::SM80_16x8x8_F16F16F16F16_TN;
};

template <>
struct MMAOpSelectorRRSM120<cute::half_t, cute::half_t, cute::half_t, 16, 8, 16> {
  using Type = cute::SM80_16x8x16_F16F16F16F16_TN;
};

template <>
struct MMAOpSelectorRRSM120<cute::half_t, cute::half_t, float, 16, 8, 8> {
  using Type = cute::SM80_16x8x8_F32F16F16F32_TN;
};

template <>
struct MMAOpSelectorRRSM120<cute::half_t, cute::half_t, float, 16, 8, 16> {
  using Type = cute::SM80_16x8x16_F32F16F16F32_TN;
};

template <>
struct MMAOpSelectorRRSM120<cute::bfloat16_t, cute::bfloat16_t, float, 16, 8, 8> {
  using Type = cute::SM80_16x8x8_F32BF16BF16F32_TN;
};

template <>
struct MMAOpSelectorRRSM120<cute::bfloat16_t, cute::bfloat16_t, float, 16, 8, 16> {
  using Type = cute::SM80_16x8x16_F32BF16BF16F32_TN;
};

}  // namespace scinthil::gemm::sm120
