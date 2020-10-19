#include "ndarray/nd_array.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace nd;
TEST(NDArrayTest, Concepts) {
  EXPECT_TRUE(contiguous_nd_storage<int>);
  EXPECT_FALSE(contiguous_nd_storage<std::ostream>);
  EXPECT_TRUE((contiguous_nd_storage<NDArray<int, 5>>));
  EXPECT_FALSE((contiguous_nd_storage<NDView<int, 5>>));

  EXPECT_TRUE((nd_object<NDView<int, 5>>));
  EXPECT_TRUE((nd_object<NDArray<int, 5>>));
  EXPECT_FALSE((nd_object<int>));
}

using namespace nd;
TEST(LazyEvaluationTest, ArrayAssignment) {
  NDArray<int, 3> A(5, 5, 5);
  int val = 0;
  for (auto& x : A)
    x = val++;

  NDArray<int, 3> B(5, 5, 5);
  val = 0;
  for (auto& x : B)
    x = 2 * val++;

  NDArray<int, 3> C = 3 * A + B;

  ASSERT_EQ(C.shape(), A.shape());

  val = 0;
  for (auto x : C)
    EXPECT_EQ(5 * val++, x);

  NDArray<int, 2> D = A(all, all, 0) + B(all, all, 1);
  ASSERT_EQ(D.shape(), (std::array<std::size_t, 2>{5, 5}));

  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j)
      EXPECT_EQ(D(i, j), A(i, j, 0) + B(i, j, 1));
}

TEST(LazyEvaluationTest, ViewAssignment) {
  auto A = zeros<int>(5, 5, 5);
  auto B = ones<int>(5, 5, 5);

  A(0, all, all) = 2 * B(all, all, 2);

  for (auto x : A(0, all, all))
    EXPECT_EQ(x, 2);
  for (auto x : A(1, all, all))
    EXPECT_EQ(x, 0);

  // Different shapes
  EXPECT_DEBUG_DEATH(A(0, range{1, end}, all) = B(all, all, 2), "Assertion.*");
}
