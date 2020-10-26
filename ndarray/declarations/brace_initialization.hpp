// Copyright (C) 2020 Giovanni Balduzzi
// All rights reserved.
//
// See LICENSE for terms of usage.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Utility for initializing an array from an initializer list.

#pragma once

#include <array>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace nd {
namespace details {

template <class T, std::size_t dims>
struct NDInitializerImpl {
  using type = std::initializer_list<typename NDInitializerImpl<T, dims - 1>::type>;
};

template <class T>
struct NDInitializerImpl<T, 0> {
  using type = T;
};

}  // namespace details

template <class T, std::size_t dims>
using NDInitializer = typename details::NDInitializerImpl<T, dims>::type;

namespace details {

template <class T, std::size_t dims>
void readData(std::vector<T>& data, std::size_t* shape,
              NDInitializer<T, dims> list) {
  if (*shape == 0) {
    *shape = list.size();
  }
  else if (*shape != list.size()) {
    throw(std::invalid_argument("List initialization from non rectangular data"));
  }

  for (const auto& elem : list) {
    if constexpr (dims == 1) {
      data.push_back(elem);
    }
    else {
      readData<T, dims - 1>(data, shape + 1, elem);
    }
  }
}

}  // namespace details

template <class T, std::size_t dims>
void readData(std::vector<T>& data, std::array<std::size_t, dims>& shape,
              NDInitializer<T, dims> list) {
  details::readData<T, dims>(data, shape.data(), list);
}

}  // namespace nd
