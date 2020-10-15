#pragma once

#include <array>
#include <numeric>
#include <cassert>

#include "broadcast.hpp"
#include "ranges.hpp"

namespace nd {

template <class T, std::size_t dims>
class NDView {
public:
  constexpr static std::size_t dimensions = dims;

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

  NDView& operator=(NDView&& rhs) = default;

  std::size_t length() const noexcept {
    return std::accumulate(shape_.begin(), shape_.end(), 1ul, std::multiplies<std::size_t>());
  }

  const auto& shape() const noexcept {
    return shape_;
  }

  template <class... Ints>
  requires is_index_pack<dims, Ints...> const T& operator()(Ints... ns) const noexcept {
    return data_[linindex(ns...)];
  }
  const T& operator()(const std::array<std::size_t, dims>& ns) const noexcept {
    return data_[linindex(ns)];
  }

  template <class... Ints>
  requires is_index_pack<dims, Ints...> T& operator()(Ints... ns) noexcept {
    return data_[linindex(ns...)];
  }
  T& operator()(const std::array<std::size_t, dims>& ns) noexcept {
    return data_[linindex(ns)];
  }

  template <class... Args>
  requires is_range<dims, Args...> auto operator()(Args... args) {
    NDView<T, range_count<Args...>> slice;
    const auto start = linindex(getStart(args)...);
    slice.data_ = data_ + start;

    unsigned i = 0;
    std::array<std::size_t, sizeof...(Args)> spans{getSpan(args, shape_[i++])...};

    unsigned target_d = 0;
    for (unsigned local_d = 0; local_d < spans.size(); ++local_d) {
      if (spans[local_d]) {
        slice.shape_[target_d] = spans[local_d];
        slice.strides_[target_d] = strides_[local_d];
        ++target_d;
      }
    }

    // TODO: debug check slice.

    return slice;
  }

private:
  template <class T2, std::size_t n2>
  friend class NDView;
  template <class T2, std::size_t n2>
  friend class NDArray;

  NDView() = default;

  template <class... Ints>
  requires is_index_pack<dims, Ints...> NDView(Ints... ns) : shape_{std::size_t(ns)...} {
    strides_[0] = 1;
    for (int i = 1; i < dims; ++i)
      strides_[i] = strides_[i - 1] * shape_[i - 1];
  }

  template <class... Ints>
  requires is_index_pack<dims, Ints...> std::size_t linindex(Ints... ids) const noexcept {
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
