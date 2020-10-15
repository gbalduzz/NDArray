#pragma once

#include <vector>

#include "nd_view.hpp"

namespace nd {

template <class T, std::size_t dims>
class NDArray {
public:
  constexpr static std::size_t dimensions = dims;

  template <class... Ints> requires is_index_pack<dims, Ints...>
  NDArray(Ints... ns) : view_{ns...} {
    data_.resize((ns * ...), 0);
    view_.data_ = data_.data();
  }

  NDArray(const NDArray& rhs){
    view_.copySize(rhs.view_);
    data_ = rhs.data_;
  }
  NDArray(NDArray&& rhs) = default;

  NDArray& operator=(const NDArray& rhs){
    view_.copySize(rhs.view_);
    data_ = rhs.data_;
    return *this;
  }
  NDArray& operator=(NDArray&& rhs) = default;

  NDArray& operator=(const T& rhs){
    std::fill(data_.begin(), data_.end(), rhs);
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

  template <class... Ints> requires is_index_pack<dims, Ints...>
  const T& operator()(Ints... ns) const noexcept {
    return view_(ns...);
  }
  const T& operator()(const std::array<std::size_t, dims>& ns) const noexcept {
    return view_(ns);
  }

  template <class... Ints> requires is_index_pack<dims, Ints...>
  T& operator()(Ints... ns) noexcept {
    return view_(ns...);
  }
  T& operator()(const std::array<std::size_t, dims>& ns) noexcept {
    return view_(ns);
  }

  template <class... Args> requires is_range<dims, Args...>
  auto operator()(Args... ns) noexcept {
    return view_(ns...);
  }

private:
  NDView<T, dims> view_;
  std::vector<T> data_;
};
}  // namespace nd
