#pragma once

#include <array>
#include <cmath>
#include <vector>

namespace nd {

template <class T>
concept contiguous_nd_storage = std::is_scalar_v<T> || (requires { T::contiguous_storage; } &&
                                                        T::contiguous_storage == true);

template <class T>
concept nd_object = requires {
  T::is_nd_object;
}
&&T::is_nd_object == true;

struct Null {};

// TODO: generalize to n arguments.
template <class F, class L, class R>
class LazyFunction {
public:
  constexpr static bool is_nd_object = true;
  constexpr static bool contiguous_storage = contiguous_nd_storage<L> && contiguous_nd_storage<R>;

  LazyFunction(F&& f, const L& l, const R& r) : f_(f), l_(l), r_(r) {}

  template <class Index>
  auto operator()(const Index& idx) const {
    if constexpr (std::is_same_v<L, Null>)
      return f_(evaluate(r_, idx));
    else if constexpr (std::is_same_v<R, Null>)
      return f_(evaluate(l_, idx));
    else
      return f_(evaluate(l_, idx), evaluate(r_, idx));
  }

  const auto& shape() const {
    if constexpr (nd_object<L> && nd_object<R>) {
      assert(l_.shape() == r_.shape());
    }
    if constexpr (nd_object<L>) {
      return l_.shape();
    }
    else {
      return r_.shape();
    }
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
  const L& l_;
  const R& r_;
};

template <class T>
concept lazy_evaluated = nd_object<T> || std::is_scalar_v<T> || std::is_same_v<T, Null>;

template <class F, lazy_evaluated L, lazy_evaluated R>
auto apply(F&& f, const L& l, const R& r) {
  return nd::LazyFunction<F, L, R>(std::forward<F>(f), l, r);
}

template <class F, lazy_evaluated L>
auto apply(F&& f, const L& l) {
  return nd::LazyFunction<F, L, Null>(std::forward<F>(f), l, Null{});
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
