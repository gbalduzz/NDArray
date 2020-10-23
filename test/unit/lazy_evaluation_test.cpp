// Copyright (C) 2020 Giovanni Balduzzi
// All rights reserved.
//
// See LICENSE for terms of usage.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Tests unary and binary operators acing on arrays and views.

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

TEST(LazyEvaluationTest, MakeTensor) {
  NDArray<double, 3> A(4, 4, 4), B(4, 4, 4);
  auto C = makeTensor(A(all, 0, all) * B(0, all, all) / 2);
  static_assert(std::is_same_v<typename decltype(C)::value_type, double>, "value type mismatch");

  EXPECT_EQ(C.shape(), (std::array<std::size_t, 2>{4, 4}));

  auto view = A(range{0, -2}, -1, -1);
  auto D = makeTensor(view);
  EXPECT_EQ(D.shape(), (std::array<std::size_t, 1>{2}));

  auto E = makeTensor(A(range{1, end}, -1, all));
  EXPECT_EQ(E.shape(), (std::array<std::size_t, 2>{3, 4}));
}
