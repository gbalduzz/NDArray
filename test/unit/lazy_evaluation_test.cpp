#include "ndarray/nd_array.hpp"

#include "gtest/gtest.h"

using namespace nd;
TEST(NDArrayTest, Plus) {
  NDArray<int, 3> A(5, 5, 5);
  int val = 0;
  for(auto& x : A) x = val++;

  NDArray<int, 3> B(5, 5, 5);
  val = 0;
  for(auto& x : B) x = 2 * val++;

  NDArray<int, 3> C(5, 5, 5);

  C =  3 * A + B;

  val = 0;
  for(auto x : C)
    EXPECT_EQ(5 * val++, x);
}
