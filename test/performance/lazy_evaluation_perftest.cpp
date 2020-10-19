#include "ndarray/nd_array.hpp"

#include <benchmark/benchmark.h>

constexpr std::size_t n = 50;
using namespace nd;

static void BM_ContiguousLazyEvaluation(benchmark::State& state) {
  nd::seed(0);
  auto A = rand<float>(n, n, n);
  auto B = rand<float>(n, n, n);
  auto C = rand<float>(n, n, n);

  for (auto _ : state) {
    C = C - A / (2. * B);
  }
}
BENCHMARK(BM_ContiguousLazyEvaluation);


static void BM_ContiguousBaselineEvaluation(benchmark::State& state) {
  nd::seed(0);
  auto A = rand<float>(n, n, n);
  auto B = rand<float>(n, n, n);
  auto C = rand<float>(n, n, n);

  for (auto _ : state) {
    for(std::size_t i = 0; i < A.size(); ++i)
      C[i] = C[i] - A[i] / (2. * B[i]);
  }
}
BENCHMARK(BM_ContiguousBaselineEvaluation);

constexpr std::size_t n2 = 200;
static void BM_NonContiguousLazyEvaluation(benchmark::State& state) {
  nd::seed(0);
  auto A = rand<float>(n2, n2, n2);
  auto B = rand<float>(n2, n2, n2);
  auto C = rand<float>(n2, n2, n2);

  for (auto _ : state) {
    C(all, 0, all) = C(all, 0, all) - A(all, 0, all) / (2. * B(all, 0, all));
  }
}
BENCHMARK(BM_NonContiguousLazyEvaluation);


static void BM_NonContiguousBaselineEvaluation(benchmark::State& state) {
  nd::seed(0);
  auto A = rand<float>(n2, n2, n2);
  auto B = rand<float>(n2, n2, n2);
  auto C = rand<float>(n2, n2, n2);

  for (auto _ : state) {
    for(std::size_t i = 0; i < n2; ++i)
      for(std::size_t j = 0; j < n2; ++j)
        C(i, 0, j) = C(i, 0, j) - A(i, 0, j) / (2. * B(i, 0, j));
  }
}
BENCHMARK(BM_NonContiguousBaselineEvaluation);
