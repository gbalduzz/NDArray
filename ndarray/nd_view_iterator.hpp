#pragma once

#include <array>

namespace nd {

template <class T, std::size_t dims>
class NDView;

template <class T, std::size_t dims, bool is_const>
class NDViewIterator : public std::iterator<std::random_access_iterator_tag, T> {
public:
  template <class U>
  using CondConst = std::conditional_t<is_const, const U, U>;
  using difference_type = std::size_t;

public:
  NDViewIterator(const NDViewIterator& rhs) = default;
  NDViewIterator& operator=(const NDViewIterator& rhs) = default;

  const T& operator*() const {
    return *(ptr_ + linindex());
  }
  T& operator*() {
    return *(ptr_ + linindex());
  }
  const T* operator->() const {
    return ptr_ + linindex();
  }
  T* operator->() {
    return ptr_ + linindex();
  }

  NDViewIterator& operator++() {
    auto& shape = *shape_;

    ++index_.back();

    for (unsigned i = dims - 1; i >= 1; --i) {
      if (index_[i] < shape[i])
        break;
      index_[i] = 0;
      ++index_[i - 1];
    }

    return (*this);
  }

  NDViewIterator& operator--() {
    auto& shape = *shape_;

    --index_.back();

    for (unsigned i = dims - 1; i >= 1; --i) {
      if (index_[i] >= 0)
        break;
      index_[i] = shape[i] - 1;
      --index_[i - 1];
    }

    return (*this);
  }

  difference_type operator-(const NDViewIterator& rhs) const {
    long int diff = 0;
    std::size_t tot_steps = 1;
    for (int i = dims - 1; i >= 0; --i) {
      diff += tot_steps * (index_[i] - rhs.index_[i]);
      tot_steps *= (*shape_)[i];
    }

    return diff;
  }

  NDViewIterator operator+(difference_type n) const {
    return NDViewIterator(*this) += n;
  }

  NDViewIterator& operator+=(difference_type n) {
    index_.back() += n;

    for (unsigned i = dims - 1; i >= 1; --i) {
      const auto carriage = index_[i] / (*shape_)[i];
      index_[i] -= carriage * (*shape_)[i];
      index_[i - 1] += carriage;
    }

    return (*this);
  }

  NDViewIterator operator-(difference_type n) const {
    return NDViewIterator(*this) -= n;
  }

  NDViewIterator& operator-=(difference_type n) {
    auto& shape = *shape_;

    index_.back() -= n;

    for (unsigned i = dims - 1; i >= 1; --i) {
      if (index_[i] < 0) {
        const auto carriage = (-index_[i] + shape[i] - 1) / shape[i];
        index_[i] += carriage * shape[i];
        index_[i - 1] -= carriage;
      }
    }

    return (*this);
  }

  auto operator==(const NDViewIterator& rhs) const noexcept {
    return index_ == rhs.index_;
  }
  auto operator!=(const NDViewIterator& rhs) const noexcept {
    return !(*this == rhs);
  }
  auto operator<(const NDViewIterator& rhs) const noexcept {
    return index_ < rhs.index_;
  }

private:
  friend class NDView<T, dims>;

  NDViewIterator(CondConst<T*> ptr, const std::array<std::size_t, dims>& shape,
                 const std::array<std::size_t, dims>& strides, const char pos)
      : ptr_(ptr), shape_(&shape), strides_(&strides) {
    index_.fill(0);
    if (pos == 'e') {
      index_[0] = shape[0];
    }
  }

  std::size_t linindex() const noexcept {
    std::size_t idx = 0;
    for (int i = 0; i < dims; ++i)
      idx += index_[i] * (*strides_)[i];
    return idx;
  }

  CondConst<T*> ptr_ = nullptr;
  std::array<long int, dims> index_;
  const std::array<std::size_t, dims>* shape_;
  const std::array<std::size_t, dims>* strides_;
};

// 1D specialization
template <class T, bool is_const>
class NDViewIterator<T, 1, is_const> : public std::iterator<std::random_access_iterator_tag, T> {
public:
  template <class U>
  using CondConst = std::conditional_t<is_const, const U, U>;
  using difference_type = std::size_t;

public:
  NDViewIterator(const NDViewIterator& rhs) = default;
  NDViewIterator& operator=(const NDViewIterator& rhs) = default;

  const T& operator*() const {
    return *(ptr_ + index_ * stride_);
  }
  T& operator*() {
    return *(ptr_ + index_ * stride_);
  }
  const T* operator->() const {
    return ptr_ + index_ * stride_;
  }
  T* operator->() {
    return ptr_ + index_ * stride_;
  }

  NDViewIterator& operator++() {
    ++index_;
    return *this;
  }

  NDViewIterator& operator--() {
    --index_;
    return *this;
  }

  difference_type operator-(const NDViewIterator& rhs) const {
    return index_ - rhs.index_;
  }

  NDViewIterator operator+(difference_type n) const {
    NDViewIterator advanced(*this);
    advanced.index_ += n;

    return advanced;
  }

  NDViewIterator operator-(difference_type n) const {
    NDViewIterator precedent(*this);
    precedent.index_ -= n;

    return precedent;
  }

  auto operator==(const NDViewIterator& rhs) const noexcept {
    return index_ == rhs.index_;
  }
  auto operator!=(const NDViewIterator& rhs) const noexcept {
    return !(*this == rhs);
  }
  auto operator<(const NDViewIterator& rhs) const noexcept {
    return index_ < rhs.index_;
  }

private:
  friend class NDView<T, 1>;

  NDViewIterator(CondConst<T*> ptr, const std::array<std::size_t, 1>& shape,
                 const std::array<std::size_t, 1>& strides, char pos)
      : ptr_(ptr), index_(pos == 'e' ? shape[0] : 0), stride_(strides[0]) {}

  CondConst<T*> ptr_ = nullptr;
  std::size_t index_ = 0;
  std::size_t stride_ = 0;
};

}  // namespace nd
