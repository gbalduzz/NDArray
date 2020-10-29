// Copyright (C) 2020 Giovanni Balduzzi
// All rights reserved.
//
// See LICENSE for terms of usage.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)

#pragma once

#include "ndarray/declarations/nd_view_iterator.hpp"

#include <array>

#include "ndarray/declarations/nd_view.hpp"

namespace nd {

template <class T, std::size_t dims, bool is_const>
NDViewIterator<T, dims, is_const>& NDViewIterator<T, dims, is_const>::operator++() {
  const auto& shape = (*view_).shape_;

  ++index_.back();

  for (unsigned i = dims - 1; i >= 1; --i) {
    if (index_[i] < shape[i])
      break;
    index_[i] = 0;
    ++index_[i - 1];
  }

  return (*this);
}

template <class T, std::size_t dims, bool is_const>
NDViewIterator<T, dims, is_const> NDViewIterator<T, dims, is_const>::operator++(int) {
  NDViewIterator cpy(*this);
  ++(*this);
  return cpy;
}

template <class T, std::size_t dims, bool is_const>
NDViewIterator<T, dims, is_const>& NDViewIterator<T, dims, is_const>::operator--() {
  const auto& shape = view_->shape_;

  --index_.back();

  for (unsigned i = dims - 1; i >= 1; --i) {
    if (index_[i] >= 0)
      break;
    index_[i] = shape[i] - 1;
    --index_[i - 1];
  }

  return (*this);
}

template <class T, std::size_t dims, bool is_const>
NDViewIterator<T, dims, is_const> NDViewIterator<T, dims, is_const>::operator--(int) {
  NDViewIterator cpy(*this);
  --(*this);
  return cpy;
}

template <class T, std::size_t dims, bool is_const>
auto NDViewIterator<T, dims, is_const>::operator-(const NDViewIterator& rhs) const -> difference_type {
  const auto& shape = view_->shape_;
  difference_type diff = 0;
  std::size_t tot_steps = 1;
  for (int i = dims - 1; i >= 0; --i) {
    diff += tot_steps * (index_[i] - rhs.index_[i]);
    tot_steps *= shape[i];
  }

  return diff;
}

template <class T, std::size_t dims, bool is_const>
NDViewIterator<T, dims, is_const>& NDViewIterator<T, dims, is_const>::operator+=(difference_type n) {
  if(n < 0) return (*this) -= - n;

  index_.back() += n;
  const auto& shape = view_->shape_;

  for (unsigned i = dims - 1; i >= 1; --i) {
    if (index_[i] < shape[i])
      break;
    const auto carriage = index_[i] / shape[i];
    index_[i] -= carriage * shape[i];
    index_[i - 1] += carriage;
  }

  return (*this);
}

template <class T, std::size_t dims, bool is_const>
NDViewIterator<T, dims, is_const>& NDViewIterator<T, dims, is_const>::operator-=(difference_type n) {
  if(n < 0) return (*this) += -n;

  const auto& shape = view_->shape_;

  index_.back() -= n;

  for (unsigned i = dims - 1; i >= 1; --i) {
    if (index_[i] >= 0)
      break;
    const auto carriage = (-index_[i] + shape[i] - 1) / shape[i];
    index_[i] += carriage * shape[i];
    index_[i - 1] -= carriage;
  }

  return (*this);
}

template <class T, std::size_t dims, bool is_const>
auto NDViewIterator<T, dims, is_const>::data() const noexcept -> CondConst<T *> {
  return view_->data_;
}

template <class T, std::size_t dims, bool is_const>
NDViewIterator<T, dims, is_const>::NDViewIterator(CondConst<NDView<T, dims>&> view, const char pos)
    : view_(&view) {
  index_.fill(0);
  if (pos == 'e') {
    index_[0] = view.shape()[0];
  }
}

template <class T, std::size_t dims, bool is_const>
std::size_t NDViewIterator<T, dims, is_const>::linindex() const noexcept {
  std::size_t idx = 0;
  for (int i = 0; i < dims; ++i)
    idx += index_[i] * view_->strides_[i];
  return idx;
}

}  // namespace nd
