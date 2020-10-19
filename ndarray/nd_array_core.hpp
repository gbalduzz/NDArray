#pragma once

#include <vector>

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

  constexpr static bool nd_object = true;
  constexpr static bool contiguous_storage = true;

  NDArray() = default;

  template <class... Ints> requires is_complete_index<dims, Ints...>
  NDArray(Ints... ns) : view_{ns...} {
    data_.resize((ns * ...), {});
    view_.data_ = data_.data();
  }

  // Constructor from compound operation.
  template <char op, class L, class R> requires contiguous_nd_storage<LazyFunction<op, L, R>>
  NDArray(const LazyFunction<op, L, R>& f) : view_(f.shape()), data_(view_.length()) {
    for (std::size_t i = 0; i < data_.size(); ++i)
      data_[i] = f(i);
  }

  NDArray(const NDArray& rhs) {
    view_.copySize(rhs.view_);
    data_ = rhs.data_;
    view_.data_ = data_.data();
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
  }

  NDArray& operator=(const T& rhs) {
    std::fill(data_.begin(), data_.end(), rhs);
    return *this;
  }

  // Assignment from compound operation.
  // Precondition: f has same shape.
  template <char op, class L, class R> requires contiguous_nd_storage<LazyFunction<op, L, R>>
  NDArray& operator=(const LazyFunction<op, L, R>& f) {
    NDArray cpy(f);
    return (*this) = std::move(cpy);
  }

  template <class... Ints>
  requires is_complete_index<dims, Ints...> void reshape(Ints... ns) {
    view_.reshape(ns...);
    data_.resize((ns * ...), {});
    view_.data_ = data_.data();
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

}  // namespace nd

template <class T, std::size_t n>
std::ostream& operator<<(std::ostream& s, const nd::NDArray<T, n>& arr) {
  return s << static_cast<nd::NDView<const T, n>>(arr);
}
