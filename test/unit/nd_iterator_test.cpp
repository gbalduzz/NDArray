#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>

#include "gtest/gtest.h"
#include <iostream>
#include <numeric>

TEST(NDIteratorTest, TwoDAccess) {
  xt::xtensor<int, 3> arr({2, 2, 2});
  std::iota(arr.rbegin(), arr.rend(), 0);

  for(auto x : arr) std::cout << x << std::endl;
//
//  auto view = arr(all, 2, all);
//  std::fill(view.begin(), view.end(), 0);
//
//  const auto last = *(--arr.end());
//  EXPECT_EQ(last, 0);
//
//  auto val = *(arr.begin());
//
//  for (int i = 0; i < arr.shape()[0]; ++i)
//    for (int j = 0; j < arr.shape()[1]; ++j)
//      for (int k = 0; k < arr.shape()[2]; ++k, --val)
//        EXPECT_EQ(arr(i, j, k), j == 2 ? 0 : val);
}
//
TEST(NDIteratorTest, TwoDSort) {
  xt::xtensor<int, 3> arr({4, 5, 5});
  std::iota(arr.rbegin(), arr.rend(), 0);

  auto v = xt::view(arr, xt::all(), 2, xt::all());
  std::cout << v << std::endl;

  ASSERT_EQ(std::distance(v.begin(), v.end()), 4 * 5);

  std::sort(v.begin(), v.end());
  std::cout << v;

  std::vector<int> values;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 5; ++j) {
      values.push_back(arr(i, 2, j));
    }

  for (int i = 0; i < values.size() - 1; ++i) {
    EXPECT_LE(values[i], values[i + 1]);
  }
}
