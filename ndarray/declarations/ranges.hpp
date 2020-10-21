// Copyright (C) 2020 Giovanni Balduzzi
// All rights reserved.
//
// See LICENSE for terms of usage.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Definition of ranges to be used in NDArray/View::operator().

#pragma once

#include <cassert>
#include <type_traits>

namespace nd {

struct range {  // [start, end)
  long int start;
  long int end;
};

struct NewAxis {};
constexpr NewAxis newaxis;

constexpr long int end = 0;
constexpr range all{0, end};

template <std::size_t N, class... Args>
constexpr bool is_complete_index = sizeof...(Args) == N &&
                                   (std::is_integral_v<std::decay_t<Args>> && ...);

template <class T>
concept axis_specifier = std::is_integral_v<std::decay_t<T>> ||
                         std::is_same_v<std::decay_t<T>, range> ||
                         std::is_same_v<std::decay_t<T>, NewAxis>;

template <std::size_t N, class... Args>
constexpr bool is_partial_index =
    !is_complete_index<N, Args...> && (axis_specifier<Args>&&...);

// free_dimensions = "original dimension" - "fixed indices" + "new axis."
template <std::size_t N, class... Args>
constexpr std::size_t free_dimensions = N -
                                        (std::size_t(std::is_integral_v<std::decay_t<Args>>) + ...) +
                                        (std::size_t(std::is_same_v<std::decay_t<Args>, NewAxis>) +
                                         ...);

namespace details {
// Returns the start of the range and advance axis_id if it belonged to an axis of the current tensor.
template <std::integral I>
std::size_t getStart(I i, const std::size_t shape) {
  return i >= 0 ? i : shape + i;
}
std::size_t getStart(const range& r, const std::size_t shape) {
  return getStart(r.start, shape);
}
std::size_t getStart(const NewAxis& /*axis*/, const std::size_t /*shape*/) {
  return 0;
}

// Returns the size of the range and advance axis_id if it belonged to an axis of the current tensor.
std::size_t getSpan(const range& r, const std::size_t shape) {
  const std::size_t idx_end = r.end > 0 ? r.end : shape + r.end;
  assert(idx_end <= shape && idx_end > r.start);
  return idx_end - r.start;
}
template <std::integral I>
std::size_t getSpan(const I idx, const std::size_t /*shape*/) {
  return 0;
}
std::size_t getSpan(const NewAxis& /*a*/, const std::size_t /*shape*/) {
  return 1;
}


template<axis_specifier Arg, std::size_t dim1, std::size_t dim2, class T>
void generateShape(const Arg& specifier,
                   const std::array<std::size_t, dim1>& old_shape,
                   const std::array<std::size_t, dim1>& old_stride,
                   std::array<std::size_t, dim2>& new_shape,
                   std::array<std::size_t, dim2>& new_stride,
                   T*& data,
                   unsigned& old_axis_id,
                   unsigned& new_axis_id) {
  const auto start = getStart(specifier, old_shape[old_axis_id]);
  data += old_stride[old_axis_id] * start;

  const auto size = getSpan(specifier, old_shape[old_axis_id]);
  if(size) {
    new_stride[new_axis_id] = old_stride[old_axis_id];
    new_shape[new_axis_id] = size;
    ++new_axis_id;
  }

  if constexpr (!std::is_same_v<std::decay_t<Arg>, NewAxis>){
    ++old_axis_id;
  }
}


}  // namespace details
}  // namespace nd
