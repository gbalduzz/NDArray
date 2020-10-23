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

namespace nd {

template <class T>
concept contiguous_nd_storage = std::is_scalar_v<T> || (requires { T::contiguous_storage; } &&
                                                        T::contiguous_storage == true);

template <class T>
concept nd_object = requires { std::decay_t<T>::is_nd_object; } &&
                    std::decay_t<T>::is_nd_object == true;

template <class T>
concept lazy_evaluated = nd_object<T> || std::is_scalar_v<T>;

template <class T>
constexpr std::size_t dimensions = 1;
template <nd_object T>
constexpr std::size_t dimensions<T> = T::dimensions;


template<std::size_t n1, std::size_t... ns>
constexpr auto pack_max = std::max(n1, pack_max<ns...>);
template<std::size_t n1>
constexpr auto pack_max<n1> = n1;

template <class F, lazy_evaluated... Args>
class LazyFunction {
public:
  constexpr static bool is_nd_object = true;
  constexpr static bool contiguous_storage = (contiguous_nd_storage<Args> && ...);
  constexpr static std::size_t dimensions = pack_max<dimensions<Args>...>;
  LazyFunction(F&& f, const Args&... args) : f_(f), args_(args...) {}

  template <class Index>
  auto operator()(const Index& idx) const {
    return invokeHelper(idx, std::make_index_sequence<sizeof...(Args)>{});
  }

  const auto& shape() const {
    return shapeHelper<0>();
  }

private:
  template <class Index, std::size_t... I>
  auto invokeHelper(const Index& idx, std::index_sequence<I...>) const {
    return f_(evaluate(std::get<I>(args_), idx)...);
  }

  template <std::size_t I>
  const auto& shapeHelper() const {
    if constexpr (nd_object<std::decay_t<std::tuple_element_t<I, Tuple>>>)
      return std::get<I>(args_).shape();
    else
      return shapeHelper<I + 1>();
  }

  template <class op2, class L2, class R2>
  const auto evaluate(const LazyFunction<op2, L2, R2>& f, std::size_t idx) const {
    return f(idx);
  }

  // TODO: collapse with previous.
  template <class F2, class L2, class R2, std::size_t dims>
  const auto evaluate(const LazyFunction<F2, L2, R2>& f,
                      const std::array<std::size_t, dims>& idx) const {
    return f(idx);
  }

  template <class T>
  requires T::contiguous_storage const auto& evaluate(const T& x, std::size_t idx) const {
    return x[idx];
  }

  template <nd_object T, std::size_t dims>
  const auto& evaluate(const T& x, const std::array<std::size_t, dims>& idx) const {
    return x(idx);
  }

  template <class T, class Idx>
  requires std::is_scalar_v<T> const auto& evaluate(const T& x, const Idx& /*idx*/) const {
    return x;
  }

  const F f_;
  using Tuple = std::tuple<const Args&...>;
  const Tuple args_;
};

template <class F, lazy_evaluated... Args>
auto apply(F&& f, const Args&... args) {
  return nd::LazyFunction<F, Args...>(std::forward<F>(f), args...);
}

template <lazy_evaluated L, lazy_evaluated R>
auto operator+(const L& l, const R& r) {
  return apply([](const auto& a, const auto& b) { return a + b; }, l, r);
}

template <lazy_evaluated L, lazy_evaluated R>
auto operator-(const L& l, const R& r) {
  return apply([](const auto& a, const auto& b) { return a - b; }, l, r);
}

template <lazy_evaluated L, lazy_evaluated R>
auto operator*(const L& l, const R& r) {
  return apply([](const auto& a, const auto& b) { return a * b; }, l, r);
}

template <lazy_evaluated L, lazy_evaluated R>
auto operator/(const L& l, const R& r) {
  return apply([](const auto& a, const auto& b) { return a / b; }, l, r);
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
