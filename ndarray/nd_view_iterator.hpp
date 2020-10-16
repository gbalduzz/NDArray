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
  NDViewIterator& operator=(const NDViewIterator& rhs) {
    index_ = rhs.index_;
    ptr_ = rhs.ptr_;
    return *this;
  }

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

  difference_type operator-(const NDViewIterator& rhs) const {
    if constexpr (dims == 1) {
      return ptr_ - rhs.ptr_;
    }
    else {
      long int diff = 0;
      std::size_t tot_steps = 1;
      for (int i = dims - 1; i >= 0; --i) {
        diff += tot_steps * (index_[i] - rhs.index_[i]);
        tot_steps *= shape_[i];
      }

      return diff;
    }
  }

  NDViewIterator operator+(difference_type n) const {
    NDViewIterator advanced(*this);

    if constexpr (dims == 1) {
      advanced.ptr_ += n * strides_[0];
    }

    else {
      advanced.index_.back() += n;
      long int tot_stride = 0;

      for (unsigned i = dims - 1; i >= 1; --i) {
        const auto carriage = advanced.index_[i] / shape_[i];
        advanced.index_[i] -= carriage * shape_[i];
        advanced.index_[i - 1] += carriage;

        tot_stride += (advanced.index_[i] - index_[i]) * strides_[i];
      }
      tot_stride += (advanced.index_[0] - index_[0]) * strides_[0];

      advanced.ptr_ += tot_stride;
    }

    return advanced;
  }

  NDViewIterator operator-(difference_type n) const {
    NDViewIterator precedent(*this);

    if constexpr (dims == 1) {
      precedent.ptr_ -= n * strides_[0];
    }

    else {
      precedent.index_.back() -= n;
      long int tot_stride = 0;

      for (unsigned i = dims - 1; i >= 1; --i) {
        const auto carriage = -precedent.index_[i] / shape_[i];
        precedent.index_[i] += carriage * shape_[i];
        precedent.index_[i - 1] -= carriage;

        tot_stride += (precedent.index_[i] - index_[i]) * strides_[i];
      }
      tot_stride += (precedent.index_[0] - index_[0]) * strides_[0];

      precedent.ptr_ += tot_stride;
    }

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
      : ptr_(ptr), shape_(shape), strides_(strides) {
    index_.fill(0);
  }

  CondConst<T*> ptr_ = nullptr;
  std::array<long int, dims> index_;
  const std::array<std::size_t, dims>& shape_;
  const std::array<std::size_t, dims>& strides_;
};

}  // namespace nd
