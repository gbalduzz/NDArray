// Copyright (C) 2020 Giovanni Balduzzi
// All rights reserved.
//
// See LICENSE for terms of usage.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Tests the usage of STL algorithms through the iterators of NDView.

#include "ndarray/nd_array.hpp"

#include "gtest/gtest.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <ranges>

using namespace nd;
namespace ranges = std::ranges;

TEST(NDIteratorTest, OneDView) {
  NDArray<int, 2> arr(4, 4);
  std::iota(arr.begin(), arr.end(), 0);

  auto view = arr(all, 1);
  // set col 1 to zero
  ranges::fill(view, 0);

  int linindex = 0;
  for (int i = 0; i < arr.shape()[0]; ++i)
    for (int j = 0; j < arr.shape()[1]; ++j, ++linindex)
      EXPECT_EQ(arr(i, j), j == 1 ? 0 : linindex);
}

using namespace nd;
TEST(NDIteratorTest, TwoDAccess) {
  NDArray<int, 3> arr(5, 5, 5);
  std::iota(arr.rbegin(), arr.rend(), 0);

  auto view = arr(all, 2, all);
  std::fill(view.begin(), view.end(), 0);

  const auto last = *(--arr.end());
  EXPECT_EQ(last, 0);

  auto val = *(arr.begin());

  for (int i = 0; i < arr.shape()[0]; ++i)
    for (int j = 0; j < arr.shape()[1]; ++j)
      for (int k = 0; k < arr.shape()[2]; ++k, --val)
        EXPECT_EQ(arr(i, j, k), j == 2 ? 0 : val);
}

TEST(NDIteratorTest, TwoDSort) {
  NDArray<int, 3> arr(4, 5, 5);
  std::iota(arr.rbegin(), arr.rend(), 0);

  auto view = arr(all, 2, all);
  static_assert(std::is_same_v<decltype(view), NDView<int, 2>>, "mismatch");
  std::cout << view << std::endl;

  ASSERT_EQ(std::distance(view.begin(), view.end()), 4 * 5);

  ranges::sort(view);
  std::cout << view;

  std::vector<int> values;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 5; ++j) {
      values.push_back(arr(i, 2, j));
    }

  for (int i = 0; i < values.size() - 1; ++i) {
    EXPECT_LE(values[i], values[i + 1]);
  }
}
