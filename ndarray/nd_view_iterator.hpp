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
    (*this) = (*this) + 1;
    return *this;
  }

  NDViewIterator& operator--() {
    (*this) = (*this) - 1;
    return *this;
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
    NDViewIterator advanced(*this);

    advanced.index_.back() += n;
    long int tot_stride = 0;

    for (unsigned i = dims - 1; i >= 1; --i) {
      const auto carriage = advanced.index_[i] / (*shape_)[i];
      advanced.index_[i] -= carriage * (*shape_)[i];
      advanced.index_[i - 1] += carriage;

      tot_stride += (advanced.index_[i] - index_[i]) * (*strides_)[i];
    }
    tot_stride += (advanced.index_[0] - index_[0]) * (*strides_)[0];

    advanced.ptr_ += tot_stride;

    return advanced;
  }

  NDViewIterator operator-(difference_type n) const {
    NDViewIterator precedent(*this);

    auto& shape = *shape_;
    auto& strides = *strides_;

    precedent.index_.back() -= n;
    long int tot_stride = 0;

    for (unsigned i = dims - 1; i >= 1; --i) {
      if (precedent.index_[i] < 0) {
        const auto carriage = (-precedent.index_[i] + shape[i] - 1) / shape[i];
        precedent.index_[i] += carriage * shape[i];
        precedent.index_[i - 1] -= carriage;
      }

      tot_stride += (precedent.index_[i] - index_[i]) * strides[i];
    }
    tot_stride += (precedent.index_[0] - index_[0]) * strides[0];

    precedent.ptr_ += tot_stride;

    return precedent;
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
      : ptr_(ptr), shape_(&shape), strides_(&strides) {
    index_.fill(0);
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

  NDViewIterator(CondConst<T*> ptr, const std::array<std::size_t, 1>& /*shape*/,
                 const std::array<std::size_t, 1>& strides)
      : ptr_(ptr), index_(0), stride_(strides[0]) {}

  CondConst<T*> ptr_ = nullptr;
  std::size_t index_ = 0;
  std::size_t stride_ = 0;
};

}  // namespace nd
