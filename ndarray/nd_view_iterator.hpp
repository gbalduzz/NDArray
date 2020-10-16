#pragma once

#include <array>

namespace nd {

template <class T, std::size_t dims>
class NDView;

template <class T, std::size_t dims, bool is_const>
class NDViewIterator {
public:
  template <class U>
  using CondConst = std::conditional_t<is_const, const U, U>;

  using iteator_category = std::bidirectional_iterator_tag;
  using value_type = T;
  using pointer = CondConst<value_type*>;
  using reference = CondConst<value_type&>;
  using difference_type = std::size_t;

public:
  const T& operator*() const {
    return *ptr_;
  }
  T& operator*() {
    return *ptr_;
  }
  const T* operator->() const {
    return ptr_;
  }
  T* operator->() {
    return ptr_;
  }

  NDViewIterator& operator++() {
    if constexpr (dims == 1) {
      ptr_ += strides_[0];
    }
    else {
      ++index_.back();
      std::size_t step = strides_.back();

      for (int i = dims - 1; i >= 1; --i) {
        if (index_[i] < shape_[i]) {
          break;
        }
        index_[i] = 0;
        ++index_[i - 1];
        step = strides_[i - 1] - (shape_[i] - 1) * strides_[i];  // TODO: update only at the end.
      }

      ptr_ += step;
    }

    return *this;
  }

  NDViewIterator& operator--() {
    if constexpr (dims == 1) {
      ptr_ -= strides_[0];
    }
    else {
      --index_.back();
      std::size_t step = strides_.back();

      for (int i = dims - 1; i >= 1; --i) {
        if (index_[i] != std::size_t(-1)) {
          break;
        }
        index_[i] = shape_[i] - 1;
        --index_[i - 1];
        step = strides_[i - 1] - (shape_[i] - 1) * strides_[i];  // TODO: update only at the end.
      }

      ptr_ -= step;
    }

    return *this;
  }

  auto operator==(const NDViewIterator& rhs) const noexcept {
    return ptr_ == rhs.ptr_;
  }
  auto operator!=(const NDViewIterator& rhs) const noexcept {
    return !(*this == rhs);
  }
  auto operator<(const NDViewIterator& rhs) const noexcept {
    return ptr_ < rhs.ptr_;
  }

private:
  friend class NDView<T, dims>;

  NDViewIterator(CondConst<T*> ptr, const std::array<std::size_t, dims>& shape,
                 const std::array<std::size_t, dims>& strides)
      : ptr_(ptr), shape_(shape), strides_(strides) {
    index_.fill(0);
  }

  CondConst<T*> ptr_ = nullptr;
  std::array<std::size_t, dims> index_;
  const std::array<std::size_t, dims>& shape_;
  const std::array<std::size_t, dims>& strides_;
};

}  // namespace nd
