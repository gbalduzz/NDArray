#include <xtensor/xtensor.hpp>

#include <iostream>
#include "gtest/gtest.h"



using namespace xt;
TEST(LazyEvaluationTest, ArrayAssignment) {
  xtensor<int, 3> A(5, 5, 5);
}
/*int val = 0;
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

TEST(LazyEvaluationTest, GenericBinaryFunction) {
  NDArray<int, 3> A(3, 3, 3), B(3, 3, 3);
  A = 1;
  B = 1;

  NDArray<std::pair<int, int>, 3> C =
      nd::apply([](auto a, auto b) { return std::make_pair(a, b); }, A * 2, B);

  for (auto x : C)
    EXPECT_EQ(x, std::make_pair(2, 1));
}

TEST(LazyEvaluationTest, GenericUnaryFunction) {
  NDArray<double, 3> A(3, 3, 3), B(3, 3, 3);
  A = 1.;

  const auto square = [](auto a) { return std::pow(a, 2); };
  B = nd::apply(square, A * 2);

  for (std::size_t i = 0; i < A.size(); ++i)
    EXPECT_DOUBLE_EQ(B[i], std::pow(2. * A[i], 2));

  B = nd::sqrt(A);

  for (std::size_t i = 0; i < A.size(); ++i)
    EXPECT_DOUBLE_EQ(B[i], std::sqrt(A[i]));
}
  */
