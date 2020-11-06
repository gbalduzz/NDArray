// Copyright (C) 2020 Giovanni Balduzzi
// All rights reserved.
//
// See LICENSE for terms of usage.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Multidimensional array of fixed dimensionality and dynamic size.

#pragma once

#include <vector>

#include "brace_initialization.hpp"
#include "lazy_functions.hpp"
#include "nd_view.hpp"

namespace nd {

template <class T, std::size_t dims>
class NDArray {
public:
  constexpr static std::size_t dimensions = dims;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;
  using value_type = T;

  constexpr static bool is_nd_object = true;
  constexpr static bool contiguous_storage = true;

  NDArray() = default;

  template <class... Ints> requires is_complete_index<dims, Ints...>
  NDArray(Ints... ns) {
    reshape(ns...);
  }

  NDArray(const std::array<std::size_t, dims>& shape) {
    reshape(shape);
  }

  void reshape(const std::array<std::size_t, dims>& shape){
    view_.reshape(shape);
    data_.resize(view_.length(), {});
    view_.data_ = data_.data();
  }

  template <class... Ints> requires is_complete_index<dims, Ints...>
  void reshape(Ints... ns){
    reshape(std::array<std::size_t, dims>{static_cast<std::size_t>(ns)...});
  }

  NDArray(NDInitializer<T, dims> elements) {
    std::array<std::size_t, dims> shape;
    shape.fill(0);

    readData(data_, shape, elements);

    view_.reshape(shape);
    view_.data_ = data_.data();
  }

  template <class F, lazy_evaluated... Args> requires (contiguous_nd_storage<LazyFunction<F, Args...>>)
  NDArray(const LazyFunction<F, Args...>& f) : view_(f.shape()), data_(view_.length()) {
    view_.data_ = data_.data();

    if(!f.broadcasted()) {
      for (std::size_t i = 0; i < data_.size(); ++i) {
        data_[i] = f[i];
      }
    }
    else {
      view_ = f;
    }
  }

  template <class F, lazy_evaluated... Args> requires (!contiguous_nd_storage<LazyFunction<F, Args...>>)
  NDArray(const LazyFunction<F, Args...>& f) : view_(f.shape()), data_(view_.length()) {
    view_.data_ = data_.data();
    view_ = f;
  }

  NDArray(const NDArray& rhs) {
    view_.copySize(rhs.view_);
    data_ = rhs.data_;
    view_.data_ = data_.data();
  }

  NDArray(const NDView<T, dims>& view) : NDArray(view.shape()) {
    view_ = view;
  }

  NDArray(NDArray&& rhs) : data_(std::move(rhs.data_)){
    view_.shallowCopy(rhs.view_);
  }

  NDArray& operator=(const NDArray& rhs) {
    view_.copySize(rhs.view_);
    data_ = rhs.data_;
    return *this;
  }

  NDArray& operator=(NDArray&& rhs) {
    view_.shallowCopy(rhs);
    data_ = std::move(rhs.data_);
    return *this;
  }

  NDArray& operator=(const T& rhs) {
    std::fill(data_.begin(), data_.end(), rhs);
    return *this;
  }

  // Assignment from compound operation.
  // Precondition: f has same shape.
  template <class F, lazy_evaluated... Args> requires (!contiguous_nd_storage<LazyFunction<F, Args...>>)
  NDArray& operator=(const LazyFunction<F, Args...>& f) {
    NDArray cpy(f);
    return (*this) = std::move(cpy);
  }
  template <class F, lazy_evaluated... Args> requires (contiguous_nd_storage<LazyFunction<F, Args...>>)
  NDArray& operator=(const LazyFunction<F, Args...>& f) {
    if(shape() != f.shape()){
      reshape(f.shape());
    }

    if(!f.broadcasted()) {
      for (std::size_t i = 0; i < data_.size(); ++i) {
        data_[i] = f[i];
      }
    }
    else{
      view_ = f;
    }

    return *this;
  }

  std::size_t length() const noexcept {
    return data_.size();
  }
  std::size_t size() const noexcept {
    return data_.size();
  }

  const auto& shape() const noexcept {
    return view_.shape_;
  }

  const T& operator[](std::size_t idx) const noexcept {
    assert(idx < data_.size());
    return data_[idx];
  }

  T& operator[](std::size_t idx) noexcept {
    assert(idx < data_.size());
    return data_[idx];
  }

  // Return reference to value
  template <class... Args>
  requires is_complete_index<dims, Args...> T& operator()(Args&&... ns) noexcept {
    return view_(std::forward<Args>(ns)...);
  }
  template <class... Args>
  requires is_complete_index<dims, Args...> const T& operator()(Args&&... ns) const noexcept {
    return view_(std::forward<Args>(ns)...);
  }

  T& operator()(const std::array<std::size_t, dims>& ns) noexcept {
    return view_(ns);
  }
  const T& operator()(const std::array<std::size_t, dims>& ns) const noexcept {
    return view_(ns);
  }

  template<std::size_t id_size>
  T& extendedElement(const std::array<std::size_t, id_size>& ns) noexcept {
    return view_.extendedElement(ns);
  }
  template<std::size_t id_size>
  const T& extendedElement(const std::array<std::size_t, id_size>& ns) const noexcept {
    return view_.extendedElement(ns);
  }

  // Return views.
  template <class... Args>
  requires is_partial_index<dims, Args...> auto operator()(Args&&... ns) noexcept {
    return view_(std::forward<Args>(ns)...);
  }
  template <class... Args>
  requires is_partial_index<dims, Args...> auto operator()(Args&&... ns) const noexcept {
    return view_(std::forward<Args>(ns)...);
  }

  operator NDView<T, dims>() noexcept {
    return view_;
  }

  operator NDView<const T, dims>() const noexcept {
    return view_;
  }

  iterator begin() noexcept {
    return data_.begin();
  }
  iterator end() noexcept {
    return data_.end();
  }
  iterator cbegin() const noexcept {
    return data_.cbegin();
  }
  iterator cend() const noexcept {
    return data_.cend();
  }

  auto rbegin() noexcept {
    return data_.rbegin();
  }
  auto rend() noexcept {
    return data_.rend();
  }

private:
  NDView<T, dims> view_;
  std::vector<T> data_;
};

template<nd_object T>
auto makeTensor(T&& view_or_func){
  constexpr std::size_t dims = std::decay_t<T>::dimensions;
  using Val = std::decay_t<decltype(view_or_func(std::array<std::size_t, dims>{}))>;
  return NDArray<Val, dims>(std::forward<T>(view_or_func));
}

template <class T, std::size_t n>
std::ostream& operator<<(std::ostream& s, const nd::NDArray<T, n>& arr) {
  return s << static_cast<nd::NDView<const T, n>>(arr);
}

}  // namespace nd
