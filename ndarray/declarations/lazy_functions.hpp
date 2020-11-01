// Copyright (C) 2020 Giovanni Balduzzi
// All rights reserved.
//
// See LICENSE for terms of usage.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Lazily evaluated compound operations on each element of arrays and views.

#pragma once

#include <array>
#include <cmath>
#include <vector>
#include <tuple>

#include "ndarray/declarations/broadcast.hpp"

namespace nd {

template <class T>
concept contiguous_nd_storage = std::is_scalar_v<T> || (requires { T::contiguous_storage; } &&
                                                        T::contiguous_storage == true);

template <class T>
concept nd_object = requires { std::decay_t<T>::is_nd_object; }
                    && std::decay_t<T>::is_nd_object == true;

template <class T>
concept lazy_evaluated = nd_object<T> || std::is_scalar_v<T>;

template<class T, std::size_t dims>
class NDArray;
template<class T>
constexpr bool is_nd_array = false;
template<class T, std::size_t dims>
constexpr bool is_nd_array<NDArray<T, dims>> = true;

template <class T>
constexpr auto getShape(const T& t) {
  return std::array<std::size_t, 0>{};
}

template <nd_object T>
const auto& getShape(const T& t) {
  return t.shape();
}

template <class T>
constexpr auto getBroadcasted(const T& t) {
  return false;
}

template <class T>
requires requires { T().broadcasted(); }
constexpr auto getBroadcasted(const T& t) {
  return t.broadcasted();
}

template <class T>
constexpr std::size_t get_dimensions = 1;

template <nd_object T>
constexpr std::size_t get_dimensions<T> = T::dimensions;

template <typename... Ts, typename F>
void for_each_in_tuple(const std::tuple<Ts...>& t, F&& f) {
  auto for_each = []<std::size_t... Is>(auto&& t, auto&& f, std::index_sequence<Is...>) {
    (std::forward<F>(f)(std::get<Is>(t)), ...);
  };

  for_each(t, std::forward<F>(f), std::make_index_sequence<sizeof...(Ts)>());
}

template <class F, lazy_evaluated... Args>
class LazyFunction {
public:
  constexpr static bool is_nd_object = true;
  constexpr static bool contiguous_storage = (contiguous_nd_storage<Args> && ...);
  constexpr static std::size_t dimensions = pack_max<get_dimensions<std::decay_t<Args>>...>;

  LazyFunction(F&& f, const Args&... args) : f_(f), args_(args...) {
    shape_.fill(0);

    for_each_in_tuple(args_, [&](const auto& arg) {
      broadcasted_ |= getBroadcasted(args_) || combineShapes(shape_, getShape(arg));
    });
  }

  bool broadcasted() const noexcept {
    return broadcasted_;
  }

  auto operator()(const std::array<std::size_t, dimensions>& idx) const {
    return invokeHelper(idx, std::make_index_sequence<sizeof...(Args)>{});
  }

  auto operator[](std::size_t idx) const {
    static_assert(contiguous_storage, "operator [] not defined for views");
    return invokeHelper(idx, std::make_index_sequence<sizeof...(Args)>{});
  }

  template <std::size_t idx_size>
  auto extendedElement(const std::array<std::size_t, idx_size>& idx) const {
    return invokeExtendedHelper(idx, std::make_index_sequence<sizeof...(Args)>{});
  }

  const auto& shape() const {
    return shape_;
  }

private:
  template <class Index, std::size_t... I>
  auto invokeHelper(const Index& idx, std::index_sequence<I...>) const {
    return f_(evaluate(std::get<I>(args_), idx)...);
  }

  template <class Index, std::size_t... I>
  auto invokeExtendedHelper(const Index& idx, std::index_sequence<I...>) const {
    return f_(evaluateExtended(std::get<I>(args_), idx)...);
  }

  template <nd_object T> requires contiguous_nd_storage<T>
  auto evaluate(const T& x, std::size_t idx) const {
    return x[idx];
  }

  template <nd_object T, class Index>
  auto evaluate(const T& x, const Index& idx) const {
    return x(idx);
  }

  template <nd_object T, class Index>
  auto evaluateExtended(const T& x, const Index& idx) const {
    return x.extendedElement(idx);
  }

  template <class T, class Idx>
  requires std::is_scalar_v<T>
  auto evaluate(const T& x, const Idx& /*idx*/) const {
    return x;
  }

  template <class T, class Idx>
  requires std::is_scalar_v<T>
  auto evaluateExtended(const T& x, const Idx& /*idx*/) const {
    return x;
  }

  // Copy views and scalars by value, NDArrays by const reference.
  template<class T>
  using LazyArgument = std::conditional_t<is_nd_array<T>, const T&, T>;

  const F f_;
  using Tuple = std::tuple<LazyArgument<Args>...>;
  const Tuple args_;
  std::array<std::size_t, dimensions> shape_;
  bool broadcasted_ = false;
};

template <class F, lazy_evaluated... Args>
auto apply(F&& f, const Args&... args) {
  return nd::LazyFunction<F, Args...>(std::forward<F>(f), args...);
}

template <lazy_evaluated L, lazy_evaluated R>
auto operator+(const L& l, const R& r) {
  return apply(std::plus<>(), l, r);
}

template <lazy_evaluated L, lazy_evaluated R>
auto operator-(const L& l, const R& r) {
  return apply(std::minus<>(), l, r);
}

template <lazy_evaluated L, lazy_evaluated R>
auto operator*(const L& l, const R& r) {
  return apply(std::multiplies<>(), l, r);
}

template <lazy_evaluated L, lazy_evaluated R>
auto operator/(const L& l, const R& r) {
  return apply(std::divides<>(), l, r);
}

template <lazy_evaluated L>
auto sqrt(const L& l) {
  return apply([](const auto& a) { return std::sqrt(a); }, l);
}

template <lazy_evaluated L, class E>
requires std::is_scalar_v<E> auto pow(const L& l, E exponent) {
  return apply([exponent](const auto& a) { return std::pow(a, exponent); }, l);
}

template <lazy_evaluated L, class E>
requires std::is_scalar_v<E> auto exp(const L& l) {
  return apply([](const auto& a) { return std::exp(a); }, l);
}

template <lazy_evaluated L, class E>
requires std::is_scalar_v<E> auto log(const L& l) {
  return apply([](const auto& a) { return std::log(a); }, l);
}

}  // namespace nd
