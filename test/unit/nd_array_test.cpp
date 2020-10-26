// Copyright (C) 2020 Giovanni Balduzzi
// All rights reserved.
//
// See LICENSE for terms of usage.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Tests the NDArray class and the access to values or views.

#include "ndarray/nd_array.hpp"

#include "gtest/gtest.h"
#include <ostream>

using namespace nd;
TEST(NDArrayTest, Shape) {
  NDArray<int, 4> arr(5, 5, 5, 10);
  EXPECT_EQ(arr.shape(), (std::array<std::size_t, 4>{5, 5, 5, 10}));
  EXPECT_EQ(arr.size(), 5 * 5 * 5 * 10);

  auto arr_view = arr(range{2, end}, 2, all, range{1, 2});
  EXPECT_EQ(arr_view.shape(), (std::array<std::size_t, 3>{3, 5, 1}));
}

TEST(NDArrayTest, ListInitialization) {
  NDArray<int, 3> arr{{{1, 2}, {3, 4}, {5, 6}}};
  EXPECT_EQ(arr.shape(), (std::array<std::size_t, 3>{1, 3, 2}));

  int val = 1;
  for (auto x : arr) {
    EXPECT_EQ(x, val++);
  }

  // Can not build with non (hyper) rectangular data.
  EXPECT_THROW((NDArray<int, 2>{{1, 2, 3}, {3, 4}}), std::invalid_argument);
}

TEST(NDArrayTest, Indexing) {
  NDArray<int, 3> arr(3, 2, 4);
  EXPECT_EQ(arr(0, 0, 1), 0);
  arr(0, 0, 0) = 1;
  EXPECT_EQ(arr[0], 1);

  EXPECT_DEBUG_DEATH(arr(3, 0, 0), "Assertion.*");
  EXPECT_DEBUG_DEATH(arr(0, range{0, 3}, 0), "Assertion.*");
  EXPECT_EQ(arr(range{0, 2}, 0, 0).length(), 2);
}

TEST(NDArrayTest, Assignment) {
  NDArray<int, 2> m(2, 2);
  m = 1;

  NDArray<int, 2> m2(2, 2);
  m2 = 2;

  m(0, all) = m2(0, all);

  std::ostringstream s;
  s << m;
  EXPECT_EQ(s.str(), "[[2, 2], [1, 1]]");

  m(all, 1) = 3;
  std::ostringstream s2;
  s2 << m(all, 1);
  EXPECT_EQ(s2.str(), "[3, 3]");

  // Negative index counts from the end.
  NDArray<int, 4> t(4, 2, 5, 6);
  auto t2 = t(range{0, -1}, -1, all, range{2, end});
  EXPECT_EQ(t2.shape(), (std::array<std::size_t, 3>{3, 5, 4}));
}

TEST(NDArrayTest, NewAxis) {
  NDArray<int, 2> m(2, 2);
  NDView<int, 3> enlarged = m(newaxis, 1, newaxis, all);
  EXPECT_EQ(enlarged.shape(), (std::array<std::size_t, 3>{1, 1, 2}));

  NDView<int, 1> view = enlarged(0, 0, all);

  std::iota(m.begin(), m.end(), 0);
  for (int j = 0; j < m.shape()[1]; ++j)
    EXPECT_EQ(m(1, j), view(j));
}

TEST(NDArrayTest, Broadcasting) {
  NDArray<int, 3> A(1, 2, 5);
  A = 1;
  NDArray<int, 3> B(1, 2, 5);
  B = 2;
  NDArray<int, 3> C(1, 2, 5);

  broadcast([](int a, int b, int& c) { c = a + b; }, A, B, C);

  // Pass and process each value of the 3 dimensional index to A.
  broadcastIndex([](int& x, const auto& idx) { x = idx[0] + idx[1] * idx[1] - idx[2]; }, A);

  for (int i = 0; i < A.shape()[0]; ++i)
    for (int j = 0; j < A.shape()[1]; ++j)
      for (int k = 0; k < A.shape()[2]; ++k)
        EXPECT_EQ(A(i, j, k), i + j * j - k);
}

TEST(NDArrayTest, TensorProduct) {
  NDArray<int, 2> A(3, 3);
  NDArray<int, 2> B(3, 3);
  NDArray<int, 4> AB(3, 3, 3, 3);

  broadcast([](auto& ab, auto a, auto b) { ab = a * b; }, AB, A(all, all, newaxis, newaxis), B);

  broadcastShape(
      [&](const auto& index) { EXPECT_EQ(AB(index), A(index[0], index[1]) * B(index[2], index[3])); },
      AB);
}

TEST(NDArrayTest, Resize) {
  NDArray<float, 3> arr;
  arr.reshape(2, 4, 1);
  EXPECT_EQ(arr.shape(), (std::array<std::size_t, 3>{2, 4, 1}));
}

TEST(NDArrayTest, InitFunctions) {
  auto arr = zeros<int>(2, 4, 5);
  EXPECT_EQ(arr.shape(), (std::array<std::size_t, 3>{2, 4, 5}));
  for (auto x : arr)
    EXPECT_EQ(0, x);

  nd::seed(42);
  auto r = rand<float>(1, 6, 3, 2);
  EXPECT_EQ(r.shape(), (std::array<std::size_t, 4>{1, 6, 3, 2}));

  std::mt19937_64 rng(42);
  std::uniform_real_distribution<float> distro(0, 1);
  for (auto x : r)
    EXPECT_EQ(distro(rng), x);
}
