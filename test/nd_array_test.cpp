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

TEST(NDArrayTest, Indexing) {
  NDArray<int, 3> arr(3, 2, 4);
  EXPECT_EQ((arr(0, 0, 0)), 0);
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
}
