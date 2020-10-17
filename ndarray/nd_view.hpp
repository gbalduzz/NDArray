#pragma once

#include <array>
#include <cassert>
#include <ostream>
#include <numeric>

#include "broadcast.hpp"
#include "ranges.hpp"
#include "nd_view_iterator.hpp"

namespace nd {

template <class T, std::size_t dims>
class NDView {
public:
  constexpr static std::size_t dimensions = dims;
  using iterator = NDViewIterator<T, dims, false>;
  using const_iterator = NDViewIterator<T, dims, true>;

  NDView(const NDView& rhs) = default;
  NDView(NDView&& rhs) = default;

  NDView& operator=(const T& rhs) {
    broadcast([=](int& a) { a = rhs; }, (*this));
    return *this;
  }

  NDView& operator=(const NDView& rhs) {
    broadcast([](int& a, int b) { a = b; }, (*this), rhs);
    return *this;
  }

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

  template <class... Args>
  requires is_partial_index<dims, Args...> auto operator()(Args... args) {
    NDView<T, free_dimensions<dims, Args...>> slice;
    const auto start = linindex(getStart(args)...);
    slice.data_ = data_ + start;

    unsigned i = 0;
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

    // TODO: debug check slice.

    return slice;
  }

  template <class... Args>
  requires is_partial_index<dims, Args...> auto operator()(Args... args) const {
    auto nonconst_view = (*const_cast<NDView<T, dims>*>(this))(args...);

    constexpr std::size_t new_dims = free_dimensions<dims, Args...>;
    return static_cast<NDView<const T, new_dims>>(nonconst_view);
  }

  operator NDView<const T, dims>() const noexcept {
    NDView<const T, dims> const_view(data_, shape_, strides_);
    return const_view;
  }

  iterator begin() noexcept {
    return iterator(data_, shape_, strides_);
  }
  const_iterator cbegin() const noexcept {
    return const_iterator(data_, shape_, strides_);
  }

  iterator end() noexcept {
    iterator it(data_, shape_, strides_);
    if constexpr (dims == 1) {
      it.index_ += shape_[0];
    }
    else {
      it.index_[0] = shape_[0];
      it.ptr_ += shape_[0] * strides_[0];
    }
    return it;
  }
  const_iterator cend() const noexcept {
    const_iterator it(nullptr, shape_, strides_);
    it.index_[0] = shape_[0];
    it.ptr_ = data_ + shape_[0] * strides_[0];
    return it;
  }

private:
  template <class T2, std::size_t n2>
  friend class NDView;
  template <class T2, std::size_t n2>
  friend class NDArray;

  NDView() = default;
  NDView(T* data, const std::array<std::size_t, dims>& shape,
         const std::array<std::size_t, dims>& strides)
      : data_(data), shape_(shape), strides_(strides) {}

  template <class... Ints>
  requires is_complete_index<dims, Ints...> NDView(Ints... ns) : shape_{std::size_t(ns)...} {
    // row-major
    strides_.back() = 1;
    for (int i = dimensions - 2; i >= 0; --i)
      strides_[i] = strides_[i + 1] * shape_[i + 1];
  }

  template <class... Ints>
      requires is_complete_index<dims, Ints...> ||
      is_partial_index<dims, Ints...> std::size_t linindex(Ints... ids) const noexcept {
    std::size_t lid = 0;
    unsigned i = 0;
    (..., (lid += ids * strides_[i++]));

    // Debug mode index check.
    assert(((ids >= 0) && ...));
    assert((i = 0, ((ids < shape_[i++]) && ...)));

    return lid;
  }

  std::size_t linindex(const std::array<std::size_t, dims>& ids) const noexcept {
    std::size_t lid = 0;
    for (std::size_t i = 0; i < ids.size(); ++i) {
      lid += ids[i] * strides_[i];
    }

    // Debug mode index check.
    // TODO
    return lid;
  }

  void copySize(const NDView& rhs) {
    shape_ = rhs.shape_;
    strides_ = rhs.strides_;
  }

  T* data_ = nullptr;
  std::array<std::size_t, dims> shape_;
  std::array<std::size_t, dims> strides_;
};

}  // namespace nd

template <class T, std::size_t n>
std::ostream& operator<<(std::ostream& s, const nd::NDView<T, n>& view) {
  s << '[';
  for (std::size_t i = 0; i < view.shape()[0]; ++i) {
    const auto slice = view(i);
    s << slice;
    if (i < view.shape()[0] - 1)
      s << ", ";
  }
  return s << ']';
}
