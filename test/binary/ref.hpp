#pragma once

#include <cmath>
#include <cstddef>
#include <span>

namespace scinthil::test::binary {

[[nodiscard]] inline bool verify_add(std::span<const float> lhs,
                                     std::span<const float> rhs,
                                     std::span<const float> out,
                                     float tolerance) {
  if (lhs.size() != rhs.size() || lhs.size() != out.size() ||
      !(tolerance >= 0.0F)) {
    return false;
  }

  for (std::size_t i{0}; i < lhs.size(); ++i) {
    const auto expected = lhs[i] + rhs[i];
    if (!(std::fabs(out[i] - expected) <= tolerance)) {
      return false;
    }
  }

  return true;
}

} // namespace scinthil::test::binary
