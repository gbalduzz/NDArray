// Copyright (C) 2020 Giovanni Balduzzi
// All rights reserved.
//
// See LICENSE for terms of usage.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)

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
  using Base = std::iterator<std::random_access_iterator_tag, T>;
  using typename Base::iterator_category;
  using typename Base::difference_type;
  using typename Base::value_type;
  using typename Base::reference;
  using typename Base::pointer;

public:
  NDViewIterator() = default;
  NDViewIterator(CondConst<NDView<T, dims>&> view, const char pos);

  NDViewIterator(const NDViewIterator& rhs) = default;
  NDViewIterator(NDViewIterator&& rhs) = default;
  NDViewIterator& operator=(const NDViewIterator& rhs) = default;
  NDViewIterator& operator=(NDViewIterator&& rhs) = default;

  const reference operator*() const {
    return *(data() + linindex());
  }
  reference operator*() {
    return *(data() + linindex());
  }
  const pointer operator->() const {
    return data() + linindex();
  }
  pointer operator->() {
    return data() + linindex();
  }
  reference operator[](std::size_t n){
    return *(*this + n);
  }
  const reference operator[](std::size_t n) const {
    return *(*this + n);
  }

  NDViewIterator& operator++();

  NDViewIterator operator++(int);

  NDViewIterator& operator--();

  NDViewIterator operator--(int);

  auto operator-(const NDViewIterator& rhs) const -> difference_type;

  NDViewIterator operator+(difference_type n) const {
    return NDViewIterator(*this) += n;
  }

  NDViewIterator& operator+=(difference_type n);

  NDViewIterator operator-(difference_type n) const {
    return NDViewIterator(*this) -= n;
  }

  NDViewIterator& operator-=(difference_type n);

  auto operator==(const NDViewIterator& rhs) const noexcept {
    return index_ == rhs.index_;
  }
  auto operator!=(const NDViewIterator& rhs) const noexcept {
    return !(*this == rhs);
  }
  auto operator<(const NDViewIterator& rhs) const noexcept {
    return index_ < rhs.index_;
  }
  auto operator<=(const NDViewIterator& rhs) const noexcept {
    return index_ <= rhs.index_;
  }
  auto operator>(const NDViewIterator& rhs) const noexcept {
    return index_ > rhs.index_;
  }
  auto operator>=(const NDViewIterator& rhs) const noexcept {
    return index_ >= rhs.index_;
  }

private:
  auto inline data() const noexcept -> CondConst<T*>;

  std::size_t linindex() const noexcept;

  std::array<long int, dims> index_;
  CondConst<NDView<T, dims>*> view_ = nullptr;
};


template <class T, std::size_t dims, bool is_const>
NDViewIterator<T, dims, is_const> operator+(
    long int n,
    const NDViewIterator<T, dims, is_const>& it) {
  return it + n;
}

template <class T, std::size_t dims, bool is_const>
NDViewIterator<T, dims, is_const> operator-(
    long int n,
    const NDViewIterator<T, dims, is_const>& it) {
  return it - n;
}

}  // namespace nd
