#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include <benchmark/benchmark.h>
#include <random>

constexpr std::size_t n = 20;
using namespace xt;

static void BM_NDArray3DSort(benchmark::State& state) {
  xtensor<std::size_t, 5> arr_5D({n, n, 3, n, 10});
  std::mt19937_64 rng(0);
  for (auto& x : arr_5D)
    x = rng();

  for (auto _ : state) {
    auto view_3D = view(arr_5D, all(), all(), 2, all(), 3);
    std::sort(view_3D.begin(), view_3D.end());
  }
}
BENCHMARK(BM_NDArray3DSort);

static void BM_3DSortBaseline(benchmark::State& state) {
  xtensor<std::size_t, 5> arr_5D({n, n, 3, n, 10});
  std::mt19937_64 rng(0);
  for (auto& x : arr_5D)
    x = rng();

  auto view_3D = view(arr_5D, all(), all(), 2, all(), 3);
  std::vector<std::size_t> linearized(view_3D.size());

  for (auto _ : state) {
    std::size_t idx = 0;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k)
          linearized[idx++] = arr_5D(i, j, 2, k, 3);

    std::sort(linearized.begin(), linearized.end());

    idx = 0;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        for (int k = 0; k < n; ++k)
          arr_5D(i, j, 2, k, 3) = linearized[idx++];
  }
}
BENCHMARK(BM_3DSortBaseline);

constexpr std::size_t n1d = n * n * n;

static void BM_NDArray1DSort(benchmark::State& state) {
  xtensor<std::size_t, 2> arr({n1d, 128});
  std::mt19937_64 rng(0);
  for (auto& x : arr)
    x = rng();

  auto v = view(arr, all(), 4);

  for (auto _ : state) {
    std::sort(v.begin(), v.end());
  }
}
BENCHMARK(BM_NDArray1DSort);

static void BM_1DSortBaseline(benchmark::State& state) {
  xtensor<std::size_t, 2> arr({n1d, 128});
  std::mt19937_64 rng(0);
  for (auto& x : arr)
    x = rng();

  auto v = view(arr, all(), 4);
  std::vector<std::size_t> linearized(v.size());

  for (auto _ : state) {
    for (int i = 0; i < n1d; ++i)
      linearized[i] = v(i);

    std::sort(linearized.begin(), linearized.end());

    for (int i = 0; i < n1d; ++i)
      v(i) = linearized[i];
  }
}
BENCHMARK(BM_1DSortBaseline);
