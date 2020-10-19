#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xview.hpp>

#include <benchmark/benchmark.h>

constexpr std::size_t n = 50;

static void BM_ContiguousLazyEvaluation(benchmark::State& state) {
  using namespace xt;
  xt::random::seed(0);
  xt::xtensor<float, 3> A = xt::random::rand<float>({n, n, n});
  xt::xtensor<float, 3> B = xt::random::rand<float>({n, n, n});
  xt::xtensor<float, 3> C = xt::random::rand<float>({n, n, n});

  for (auto _ : state) {
    C = C - A / (2. * B);
  }
}
BENCHMARK(BM_ContiguousLazyEvaluation);

static void BM_ContiguousBaselineEvaluation(benchmark::State& state) {
  std::mt19937_64 rng(0);
  std::uniform_real_distribution<float> distro(0, 1);

  std::vector<float> A(n * n * n), B(n * n * n), C(n * n * n);
  for (auto& x : A)
    x = distro(rng);
  for (auto& x : B)
    x = distro(rng);
  for (auto& x : C)
    x = distro(rng);

  for (auto _ : state) {
    for (std::size_t i = 0; i < C.size(); ++i)
      C[i] = C[i] - A[i] / (2. * B[i]);
  }
}
BENCHMARK(BM_ContiguousBaselineEvaluation);
//
constexpr std::size_t n2 = 200;
static void BM_NonContiguousLazyEvaluation(benchmark::State& state) {
  xt::random::seed(0);
  xt::xtensor<float, 3> A = xt::random::rand<float>({n2, n2, n2});
  xt::xtensor<float, 3> B = xt::random::rand<float>({n2, n2, n2});
  xt::xtensor<float, 3> C = xt::random::rand<float>({n2, n2, n2});

  using namespace xt;

  for (auto _ : state) {
    view(C, all(), 0, all()) =
        view(C, all(), 0, all()) - view(A, all(), 0, all()) / (2. * view(B, all(), 0, all()));
  }
}
BENCHMARK(BM_NonContiguousLazyEvaluation);
