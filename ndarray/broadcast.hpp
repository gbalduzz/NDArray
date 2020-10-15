#pragma once

#include <array>
#include <utility>

namespace nd {

namespace {
template <std::size_t i, std::size_t dims>
struct Iterate {
  template <class F, class View1, class... Views>
  static void execute(std::array<std::size_t, dims>& index, F&& f, View1&& view1, Views&&... views) {
    for (index[i] = 0; index[i] < view1.shape()[i]; ++index[i]) {
      Iterate<i + 1, dims>::execute(index, std::forward<F>(f), std::forward<View1>(view1),
                                    std::forward<Views>(views)...);
    }
  }
};

template <std::size_t dims>
struct Iterate<0, dims> {
  template <class F, class View1, class... Views>
  static void execute(F&& f, View1&& view1, Views&&... views) {
    std::array<std::size_t, dims> index;

    for (index[0] = 0; index[0] < view1.shape()[0]; ++index[0]) {
      Iterate<1, dims>::execute(index, std::forward<F>(f), std::forward<View1>(view1),
                                std::forward<Views>(views)...);
    }
  }
};

template <std::size_t dims>
struct Iterate<dims, dims> {
  template <class F, class... Views>
  static void execute(const std::array<std::size_t, dims>& index, F&& f, Views&&... views) {
    std::forward<F>(f)(std::forward<Views>(views)(index)...);
  }
};

}  // namespace

template <class F, class View1, class... Views>
void broadcast(F&& f, View1&& view1, Views&&... views) {
  constexpr std::size_t dims = std::decay_t<View1>::dimensions;

  static_assert(((dims == std::decay_t<Views>::dimensions) && ...),
                "Number of dimensions mismatch in broadcast operation");
  assert(((view1.shape() == views.shape()) && ...));

  Iterate<0, dims>::execute(std::forward<F>(f), std::forward<View1>(view1),
                            std::forward<Views>(views)...);
}

}  // namespace nd
