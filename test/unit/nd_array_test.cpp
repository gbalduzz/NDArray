#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include "gtest/gtest.h"
#include <ostream>

TEST(NDArrayTest, Broadcasting) {
  using namespace xt;

  xt::random::seed(0);
  const int n = 3;
  xt::xtensor<float, 3> A = xt::random::rand<float>({n, n, n});
  xt::xtensor<float, 3> B = xt::random::rand<float>({n, n, n});
  xt::xtensor<float, 3> C = xt::random::rand<float>({n, n, n});
  xt::xtensor<float, 2> result;

  result = view(C, all(), 0, all()) - view(A, all(), 0, all()) / (2. * view(B, all(), 0, all()));

  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      EXPECT_EQ(result(0, 0, 0), C(0, 0, 0) - A(0, 0, 0) / (2 * B(0, 0, 0)));
}
