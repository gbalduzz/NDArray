// Copyright (C) 2020 Giovanni Balduzzi
// All rights reserved.
//
// See LICENSE for terms of usage.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Free functions to initialize an NDArray with specific values.

#pragma once

#include <type_traits>
#include <random>

#include "ndarray/declarations/nd_array.hpp"

namespace nd {

template <class T, std::integral... Is>
auto zeros(Is... shape) {
  nd::NDArray<T, sizeof...(Is)> arr(shape...);
  arr = T{0};
  return arr;
}

template <class T, std::integral... Is>
auto ones(Is... shape) {
  nd::NDArray<T, sizeof...(Is)> arr(shape...);
  arr = T{1};
  return arr;
}

std::mt19937_64 __rng;

void seed(std::size_t s) {
  __rng.seed(s);
}

template <std::floating_point T, std::integral... Is>
auto rand(Is... shape) {
  std::uniform_real_distribution<T> distro(0, 1.);
  nd::NDArray<T, sizeof...(Is)> arr(shape...);
  for (auto& x : arr)
    x = distro(__rng);
  return arr;
}

template <std::integral T, std::integral... Is>
auto rand(Is... shape) {
  nd::NDArray<T, sizeof...(Is)> arr(shape...);
  for (auto& x : arr)
    x = __rng();
  return arr;
}

}  // namespace nd
