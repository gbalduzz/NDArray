// Copyright (C) 2020 Giovanni Balduzzi
// All rights reserved.
//
// See LICENSE for terms of usage.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Non-owning view of an NDArray or another view.

#pragma once

#include "ndarray/declarations/nd_view.hpp"


#include <numeric>

#include "ndarray/declarations/nd_view_iterator.hpp"

namespace nd {

template <class T, std::size_t dims>
template<class F, class L, class R>
NDView<T, dims>& NDView<T, dims>::operator=(const LazyFunction<F, L, R>& f){
  assert(shape() == f.shape());
  broadcastIndex([&](auto& x, const auto& index){x = f(index); }, *this);
  return *this;
}

template <class T, std::size_t dims>
NDView<T, dims>& NDView<T, dims>::shallowCopy(const NDView& rhs){
  data_ = rhs.data_;
  shape_ = rhs.shape_;
  strides_ = rhs.strides_;
  return *this;
}

template <class T, std::size_t dims>
template <class... Args> requires is_partial_index<dims, Args...>
auto NDView<T, dims>::operator()(Args... args) {
  NDView<T, free_dimensions<dims, Args...>> slice;

  unsigned i = 0;
  std::array<std::size_t, sizeof...(Args)> start{getStart(args, shape_[i++])...};
  slice.data_ = data_ + linindex(start);

  i = 0;
  std::array<std::size_t, sizeof...(Args)> spans{getSpan(args, shape_[i++])...};

  unsigned target_d = 0;
  unsigned local_d = 0;
  for (; local_d < spans.size(); ++local_d) {
    if (spans[local_d]) {
      slice.shape_[target_d] = spans[local_d];
      slice.strides_[target_d] = strides_[local_d];
      ++target_d;
    }
  }

  // Fill missing dimensions with complete range
  for (; local_d < dimensions; ++local_d) {
    slice.shape_[target_d] = shape_[local_d];
    slice.strides_[target_d] = strides_[local_d];
    ++target_d;
  }

  // TODO: check slice in debug mode.
  return slice;
}


template <class T, std::size_t dims>
auto NDView<T, dims>::begin() noexcept -> iterator {
  return iterator(*this, 'b');
}
template <class T, std::size_t dims>
auto NDView<T, dims>::begin() const noexcept -> const_iterator {
  return iterator(*this, 'b');
}
template <class T, std::size_t dims>
auto NDView<T, dims>::cbegin() const noexcept -> const_iterator {
  return iterator(*this, 'b');
}

template <class T, std::size_t dims>
auto NDView<T, dims>::end() noexcept -> iterator {
  return iterator(*this, 'e');
}
template <class T, std::size_t dims>
auto NDView<T, dims>::end() const noexcept -> const_iterator {
  return iterator(*this, 'e');
}
template <class T, std::size_t dims>
auto NDView<T, dims>::cend() const noexcept -> const_iterator {
  return iterator(*this, 'e');
}

template <class T, std::size_t dims>
NDView<T, dims>::NDView(const std::array<std::size_t, dims>& ns) : shape_(ns)  {
  strides_.back() = 1;
  for (int i = dimensions - 2; i >= 0; --i)
    strides_[i] = strides_[i + 1] * shape_[i + 1];
}

template <class T, std::size_t dims>
template <class... Ints> requires is_complete_index<dims, Ints...>
void NDView<T, dims>::reshape(Ints... ns) {
  shape_ = {std::size_t(ns)...};
  strides_.back() = 1;
  for (int i = dimensions - 2; i >= 0; --i)
    strides_[i] = strides_[i + 1] * shape_[i + 1];
}

template <class T, std::size_t dims>
template <class... Ints> requires is_complete_index<dims, Ints...> ||
                                  is_partial_index<dims, Ints...>
std::size_t NDView<T, dims>::linindex(Ints... ids) const noexcept {
  std::size_t lid = 0;
  unsigned i = 0;
  (..., (lid += ids * strides_[i++]));

  // Debug mode index check.
  assert(((ids >= 0) && ...));
  assert((i = 0, ((ids < shape_[i++]) && ...)));

  return lid;
}

template <class T, std::size_t dims>
template<std::size_t id_size> requires (id_size <= dims)
std::size_t NDView<T, dims>::linindex(const std::array<std::size_t, id_size>& ids) const noexcept {
  std::size_t lid = 0;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    assert(ids[i] < shape_[i]);
    lid += ids[i] * strides_[i];
  }

  return lid;
}

template <class T, std::size_t dims>
void NDView<T, dims>::copySize(const NDView& rhs) {
  shape_ = rhs.shape_;
  strides_ = rhs.strides_;
}


template <class T, std::size_t dims>
std::ostream& operator<<(std::ostream& s, const NDView<T, dims>& view) {
  s << '[';
  for (std::size_t i = 0; i < view.shape()[0]; ++i) {
    const auto slice = view(i);
    s << slice;
    if (i < view.shape()[0] - 1)
      s << ", ";
  }
  return s << ']';
}

}  // namespace nd
