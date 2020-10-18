#pragma once

#include <array>
#include <cassert>
#include <utility>

namespace nd {

namespace {
template <std::size_t i, std::size_t dims, bool pass_index>
struct Iterate {
  template <class F, class... Views>
  static void execute(std::array<std::size_t, dims>& index,
                      const std::array<std::size_t, dims>& shape, F&& f, Views&&... views) {
    for (index[i] = 0; index[i] < shape[i]; ++index[i]) {
      Iterate<i + 1, dims, pass_index>::execute(index, shape, std::forward<F>(f),
                                                std::forward<Views>(views)...);
    }
  }
};

template <std::size_t dims, bool pass_index>
struct Iterate<dims, dims, pass_index> {
  template <class F, class... Views>
  static void execute(const std::array<std::size_t, dims>& index,
                      const std::array<std::size_t, dims>& /*shape*/, F&& f, Views&&... views) {
    if constexpr (pass_index) {
      std::forward<F>(f)(std::forward<Views>(views)(index)..., index);
    }
    else {
      std::forward<F>(f)(std::forward<Views>(views)(index)...);
    }
  }
};

template <bool pass_index, class F, class View1, class... Views>
void broadcastImpl(F&& f, View1&& view1, Views&&... views) {
  constexpr std::size_t dims = std::decay_t<View1>::dimensions;

  static_assert(((dims == std::decay_t<Views>::dimensions) && ...),
                "Number of dimensions mismatch in broadcast operation");
  assert(((view1.shape() == views.shape()) && ...));

  std::array<std::size_t, dims> index;
  index.fill(0);

  Iterate<0, dims, pass_index>::execute(index, view1.shape(), std::forward<F>(f),
                                        std::forward<View1>(view1), std::forward<Views>(views)...);
}

}  // namespace

template <class F, class... Views>
void broadcast(F&& f, Views&&... views) {
  broadcastImpl<false>(std::forward<F>(f), std::forward<Views>(views)...);
}

template <class F, class... Views>
void broadcastIndex(F&& f, Views&&... views) {
  broadcastImpl<true>(std::forward<F>(f), std::forward<Views>(views)...);
}

}  // namespace nd
