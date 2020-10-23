// Copyright (C) 2020 Giovanni Balduzzi
// All rights reserved.
//
// See LICENSE for terms of usage.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Performance test of lazily evaluated compound operations. See the git branch 'xtensor_test' for
// the equivalent test on xtensors.

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
  std::mt19937_64 rng(0);
  std::uniform_real_distribution<float> distro(0, 1);

  std::vector<float> A(n * n * n), B(n * n * n), C(n * n * n);
  for (std::size_t i = 0; i < A.size(); ++i) {
    A[i] = distro(rng);
    B[i] = distro(rng);
    C[i] = distro(rng);
  }

  for (auto&& _ : state) {
    for (std::size_t i = 0; i < C.size(); ++i)
      C[i] = C[i] - A[i] / (2. * B[i]);
  }
}

BENCHMARK(BM_ContiguousBaselineEvaluation);

static void BM_NonContiguousLazyEvaluation(benchmark::State& state) {
  nd::seed(0);
  auto A = rand<float>(n, n, 5ul, n, 4ul);
  auto B = rand<float>(n, n, 5ul, n, 4ul);
  auto C = rand<float>(n, n, 5ul, n, 4ul);
  auto res = NDArray<float, 3>(n, n, n);

  for (auto _ : state) {
    res = C(all, all, 0, all, 0) - A(all, all, 0, all, 0) / (2. * B(all, all, 0, all, 0));
  }
}
BENCHMARK(BM_NonContiguousLazyEvaluation);

static void BM_BroadcastingLazyEvaluation(benchmark::State& state) {
  nd::seed(0);
  auto A = rand<float>(1, n, 5ul, n, 4ul);
  auto B = rand<float>(n, 1, 5ul, n, 4ul);
  auto C = rand<float>(n, n, 5ul, 1, 4ul);
  auto res = NDArray<float, 3>(n, n, n);

  for (auto _ : state) {
    res = C(all, all, 0, all, 0) - A(all, all, 0, all, 0) / (2. * B(all, all, 0, all, 0));
  }
}
BENCHMARK(BM_BroadcastingLazyEvaluation);
