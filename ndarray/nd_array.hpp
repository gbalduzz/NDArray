#pragma once

#include <vector>

#include "nd_view.hpp"

namespace nd {

template <class T, std::size_t dims>
class NDArray {
public:
  constexpr static std::size_t dimensions = dims;
  using iterator = typename std::vector<T>::iterator;
  using const_iterator = typename std::vector<T>::const_iterator;

  template <class... Ints>
  requires is_complete_index<dims, Ints...> NDArray(Ints... ns) : view_{ns...} {
    data_.resize((ns * ...), 0);
    view_.data_ = data_.data();
  }

  NDArray(const NDArray& rhs) {
    view_.copySize(rhs.view_);
    data_ = rhs.data_;
  }
  NDArray(NDArray&& rhs) = default;

  NDArray& operator=(const NDArray& rhs) {
    view_.copySize(rhs.view_);
    data_ = rhs.data_;
    return *this;
  }
  NDArray& operator=(NDArray&& rhs) = default;

  NDArray& operator=(const T& rhs) {
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

  // Return reference to value
  template <class... Args>
  requires is_complete_index<dims, Args...> T& operator()(Args&&... ns) noexcept {
    return view_(std::forward<Args>(ns)...);
  }
  template <class... Args>
  requires is_complete_index<dims, Args...> const T& operator()(Args&&... ns) const noexcept {
    return view_(std::forward<Args>(ns)...);
  }

  // TODO: remove
  T& operator()(const std::array<std::size_t, dims>& ns) noexcept {
    return view_(ns);
  }
  const T& operator()(const std::array<std::size_t, dims>& ns) const noexcept {
    return view_(ns);
  }

  // Return views.
  template <class... Args>
  auto operator()(Args&&... ns) noexcept {
    return view_(std::forward<Args>(ns)...);
  }
  template <class... Args>
  auto operator()(Args&&... ns) const noexcept {
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

}  // namespace nd

template <class T, std::size_t n>
std::ostream& operator<<(std::ostream& s, const nd::NDArray<T, n>& arr) {
  return s << static_cast<nd::NDView<const T, n>>(arr);
}
