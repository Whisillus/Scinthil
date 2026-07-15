#include <cutlass/numeric_types.h>
#include <gtest/gtest.h>

#include "test_binary_testbed.hpp"

namespace {

struct Add {
  template <typename Dtype>
  __host__ __device__ Dtype operator()(Dtype lhs, Dtype rhs) const {
    return lhs + rhs;
  }
};

}  // namespace

TEST(ScinthilBinaryTest, TestAdd) {
  scinthil::test::binary::test_binary<float, Add>(333);
  scinthil::test::binary::test_binary<float, Add>(1024);
  scinthil::test::binary::test_binary<float, Add>(2222);

  scinthil::test::binary::test_binary<cutlass::half_t, Add>(333);
  scinthil::test::binary::test_binary<cutlass::half_t, Add>(1024);
  scinthil::test::binary::test_binary<cutlass::half_t, Add>(2222);

  scinthil::test::binary::test_binary<cutlass::bfloat16_t, Add>(333);
  scinthil::test::binary::test_binary<cutlass::bfloat16_t, Add>(1024);
  scinthil::test::binary::test_binary<cutlass::bfloat16_t, Add>(2222);
}
