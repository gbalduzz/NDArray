// Copyright (C) 2020 Giovanni Balduzzi
// All rights reserved.
//
// See LICENSE for terms of usage.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Non-owning view of an NDArray or another view.

#pragma once

#include <array>
#include <cassert>
#include <ostream>
#include <numeric>
#include <ranges>
#include <tuple>

#include "ndarray/declarations/broadcast.hpp"
#include "ndarray/declarations/lazy_functions.hpp"
#include "ndarray/declarations/ranges.hpp"

namespace nd {

template <class T, std::size_t n, bool is_const>
class NDViewIterator;

template <class T, std::size_t dims>
class NDView {
public:
  constexpr static std::size_t dimensions = dims;
  using iterator = NDViewIterator<T, dims, false>;
  using const_iterator = NDViewIterator<T, dims, true>;
  using value_type = T;

  constexpr static bool is_nd_object = true;
  constexpr static bool contiguous_storage = false;

  NDView(const NDView& rhs) = default;
  NDView(NDView&& rhs) = default;

  NDView& operator=(const T& rhs) {
    broadcast([=](int& a) { a = rhs; }, (*this));
    return *this;
  }

  NDView& operator=(const NDView& rhs) {
    broadcast([](T& a, T b) { a = b; }, (*this), rhs);
    return *this;
  }

  template<class F, class... Args>
  NDView& operator=(const LazyFunction<F, Args...>& f);

  NDView& shallowCopy(const NDView& rhs);

  std::size_t length() const noexcept {
    return std::accumulate(shape_.begin(), shape_.end(), 1ul, std::multiplies<std::size_t>());
  }

  const auto& shape() const noexcept {
    return shape_;
  }

  template <class... Ints>
  requires is_complete_index<dims, Ints...> const T& operator()(Ints... ns) const noexcept {
    return data_[linindex(ns...)];
  }
  const T& operator()(const std::array<std::size_t, dims>& ns) const noexcept {
    return data_[linindex(ns)];
  }

  template <class... Ints>
  requires is_complete_index<dims, Ints...> T& operator()(Ints... ns) noexcept {
    return data_[linindex(ns...)];
  }
  T& operator()(const std::array<std::size_t, dims>& ns) noexcept {
    return data_[linindex(ns)];
  }

  template <class... Args> requires is_partial_index<dims, Args...>
  auto operator()(Args... args);

  template <class... Args>
  requires is_partial_index<dims, Args...> auto operator()(Args... args) const {
    auto nonconst_view = (*const_cast<NDView<T, dims>*>(this))(args...);
    constexpr std::size_t new_dims = decltype(nonconst_view)::dimensions;

    return static_cast<NDView<const T, new_dims>>(nonconst_view);
  }

  operator NDView<const T, dims>() const noexcept {
    NDView<const T, dims> const_view(data_, shape_, strides_);
    return const_view;
  }

  // Access methods for broadcasting operations. Dimensions with size 1 or past the end are ignored.
  template <std::size_t id_dim>
  const T& extendedElement(const std::array<std::size_t, id_dim>& index) const noexcept {
    return data_[linindexExtended(index)];
  }
  template <std::size_t id_dim>
  T& extendedElement(const std::array<std::size_t, id_dim>& index) noexcept {
    return data_[linindexExtended(index)];
  }
  auto inline begin() noexcept -> iterator;
  auto inline begin() const noexcept -> const_iterator;
  auto inline cbegin() const noexcept -> const_iterator;

  auto inline end() noexcept -> iterator;
  auto inline end() const noexcept -> const_iterator;
  auto inline cend() const noexcept -> const_iterator;

private:
  template <class T2, std::size_t n2>
  friend class NDView;
  template <class T2, std::size_t n2, bool is_const>
  friend class NDViewIterator;
  template <class T2, std::size_t n2>
  friend class NDArray;

  NDView() = default;
  NDView(T* data, const std::array<std::size_t, dims>& shape,
         const std::array<std::size_t, dims>& strides)
      : data_(data), shape_(shape), strides_(strides) {}

  template <class... Ints> requires is_complete_index<dims, Ints...>
  NDView(Ints... ns) {
    reshape(ns...);
  }

  NDView(const std::array<std::size_t, dims>& ns);

  template <class... Ints> requires is_complete_index<dims, Ints...>
  void reshape(Ints... ns);
  void reshape(const std::array<std::size_t, dims>& ns);

  template <class... Ints> requires is_complete_index<dims, Ints...> ||
                                    is_partial_index<dims, Ints...> std::size_t
  linindex(Ints... ids) const noexcept;

  template<std::size_t id_size> requires (id_size <= dims)
  std::size_t linindex(const std::array<std::size_t, id_size>& ids) const noexcept;

  template<std::size_t id_size> requires (id_size >= dims)
  std::size_t linindexExtended(const std::array<std::size_t, id_size>& ids) const noexcept;

  void copySize(const NDView& rhs);

  T* data_ = nullptr;
  std::array<std::size_t, dims> shape_;
  std::array<std::size_t, dims> strides_;
};

template <class T, std::size_t n>
std::ostream& operator<<(std::ostream& s, const nd::NDView<T, n>& view);

}  // namespace nd

// The iterator has a pointer to a view. Dangling is not safe.
template<class T, std::size_t n>
inline constexpr bool std::ranges::enable_borrowed_range<nd::NDView<T, n>> = false;
