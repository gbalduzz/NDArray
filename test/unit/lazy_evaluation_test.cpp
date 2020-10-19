#include "ndarray/nd_array.hpp"

#include <iostream>
#include "gtest/gtest.h"

using namespace nd;
TEST(NDArrayTest, Concepts) {
  EXPECT_TRUE(contiguous_nd_storage<int>);
  EXPECT_FALSE(contiguous_nd_storage<std::ostream>);
  EXPECT_TRUE((contiguous_nd_storage<NDArray<int, 5>>));
  EXPECT_FALSE((contiguous_nd_storage<NDView<int, 5>>));
}

using namespace nd;
TEST(NDArrayTest, Plus) {
  NDArray<int, 3> A(5, 5, 5);
  int val = 0;
  for(auto& x : A) x = val++;

  NDArray<int, 3> B(5, 5, 5);
  val = 0;
  for(auto& x : B) x = 2 * val++;

  NDArray<int, 3> C =  3 * A + B;

  ASSERT_EQ(C.shape(), A.shape());

  val = 0;
  for(auto x : C)
    EXPECT_EQ(5 * val++, x);
}
