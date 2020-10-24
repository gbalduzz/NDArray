// Copyright (C) 2020 Giovanni Balduzzi
// All rights reserved.
//
// See LICENSE for terms of usage.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Functors to loop through each NDArray/View element.

#pragma once

#include <array>
#include <cassert>
#include <stdexcept>
#include <utility>

namespace nd {

template <std::size_t I1, std::size_t... Is>
constexpr std::size_t pack_max = std::max(I1, pack_max<Is...>);
template <std::size_t I>
constexpr std::size_t pack_max<I> = I;

template <std::size_t I1, std::size_t... Is>
constexpr bool pack_equal = ((I1 == Is) && ...);

template <std::size_t n1, std::size_t n2> requires(n1 >= n2)
bool combineShapes(std::array<std::size_t, n1>& s1,
                   const std::array<std::size_t, n2>& s2) {
  if constexpr (n2 == 0){ // Scalar needs not broadcast an index.
    return false;
  }
  constexpr std::size_t shift = n1 - n2;
  bool broadcasted = n1 != n2;
  for (unsigned i = 0; i < n2; ++i) {
    assert(s1[i+shift] == s2[i] || s1[i+shift] == 0 || s1[i+shift] == 1 || s2[i] == 1);
    s1[i + shift] = std::max(s1[i + shift], s2[i]);
    broadcasted |= s1[i + shift] != 0 && s1[i + shift] != s2[i];
  }

  return broadcasted;
}

// Returns a pair of the resulting shape and a boolean value set to true if a broadcast took place
template <std::size_t... I>
auto getBroadcastShape(const std::array<std::size_t, I>&... shapes) {
  std::array<std::size_t, pack_max<I...>> result_shape;
  result_shape.fill(0);

  const bool bcasted = (combineShapes(result_shape, shapes) ||...);
  return std::make_pair(result_shape, bcasted);
}

enum struct BroadcastMode {
  tensors,
  index,
  tensors_and_index,
  tensors_extended,
  tensors_and_index_extended
};

namespace {
template <std::size_t i, std::size_t dims, BroadcastMode mode>
struct Iterate {
  template <class F, class... Views>
  static void execute(std::array<std::size_t, dims>& index,
                      const std::array<std::size_t, dims>& shape, F&& f, Views&&... views) {
    for (index[i] = 0; index[i] < shape[i]; ++index[i]) {
      Iterate<i + 1, dims, mode>::execute(index, shape, std::forward<F>(f),
                                          std::forward<Views>(views)...);
    }
  }
};

template <std::size_t dims, BroadcastMode mode>
struct Iterate<dims, dims, mode> {
  template <class F, class... Views>
  static void execute(const std::array<std::size_t, dims>& index,
                      const std::array<std::size_t, dims>& /*shape*/, F&& f, Views&&... views) {
    if constexpr (mode == BroadcastMode::tensors_and_index) {
      std::forward<F>(f)(std::forward<Views>(views)(index)..., index);
    }
    else if constexpr (mode == BroadcastMode::tensors_and_index_extended) {
      std::forward<F>(f)(std::forward<Views>(views).extendedElement(index)..., index);
    }
    else if constexpr (mode == BroadcastMode::tensors) {
      std::forward<F>(f)(std::forward<Views>(views)(index)...);
    }
    else if constexpr (mode == BroadcastMode::tensors_extended) {
      std::forward<F>(f)(std::forward<Views>(views).extendedElement(index)...);
    }
    else if constexpr (mode == BroadcastMode::index) {
      std::forward<F>(f)(index);
    }
  }
};

}  // namespace

template <class F, class... Views>
void broadcast(F&& f, Views&&... views) {
  const auto [shape, broadcast] = getBroadcastShape(views.shape()...);

  constexpr std::size_t dims = pack_max<std::decay_t<Views>::dimensions...>;
  std::array<std::size_t, dims> index;
  index.fill(0);

  if constexpr (pack_equal<std::decay_t<Views>::dimensions...>) {
    if (!broadcast) {
      Iterate<0, dims, BroadcastMode::tensors>::execute(index, shape, std::forward<F>(f),
                                                        std::forward<Views>(views)...);
    }
  }
  if (broadcast) {
    Iterate<0, dims, BroadcastMode::tensors_extended>::execute(index, shape, std::forward<F>(f),
                                                               std::forward<Views>(views)...);
  }
}

template <class F, class... Views>
void broadcastIndex(F&& f, Views&&... views) {
  auto [shape, broadcast] = getBroadcastShape(views.shape()...);

  decltype(shape) index;
  constexpr std::size_t dims = index.size();
  index.fill(0);

  if constexpr (pack_equal<std::decay_t<Views>::dimensions...>) {
    if (!broadcast) {
      Iterate<0, dims, BroadcastMode::tensors_and_index>::execute(index, shape, std::forward<F>(f),
                                                                  std::forward<Views>(views)...);
    }
  }
  if (broadcast) {
    Iterate<0, dims, BroadcastMode::tensors_and_index_extended>::execute(
        index, shape, std::forward<F>(f), std::forward<Views>(views)...);
  }
}

template <class F, std::size_t dims>
void broadcastShape(F&& f, const std::array<std::size_t, dims>& shape) {
  std::array<std::size_t, dims> index;
  index.fill(0);

  Iterate<0, dims, BroadcastMode::index>::execute(index, shape, std::forward<F>(f));
}

template <class F, class T>
void broadcastShape(F&& f, const T& tensor) {
  broadcastShape(std::forward<F>(f), tensor.shape());
}


}  // namespace nd
