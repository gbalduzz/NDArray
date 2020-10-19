#pragma once

#include <vector>
#include <array>

namespace nd {

template <class T>
concept contiguous_nd_storage = std::is_scalar_v<T> ||
                                (requires { T::contiguous_storage; } && T::contiguous_storage == true);

template <class T>
concept nd_object = requires { T::is_nd_object; } && T::is_nd_object == true;

template <char op, class L, class R>
class LazyFunction {
public:
  constexpr static bool is_nd_object = true;
  constexpr static bool contiguous_storage = contiguous_nd_storage<L> && contiguous_nd_storage<R>;

  LazyFunction(const L& l, const R& r) : l_(l), r_(r) {}

  template<class Index>
  auto operator()(const Index& idx) const {
    if constexpr (op == '+')
      return evaluate(l_, idx) + evaluate(r_, idx);
    else if constexpr (op == '-')
      return evaluate(l_, idx) - evaluate(r_, idx);
    else if constexpr (op == '*')
      return evaluate(l_, idx) * evaluate(r_, idx);
    else if constexpr (op == '/')
      return evaluate(l_, idx) / evaluate(r_, idx);
    else
      throw("Not supported");
  }

  const auto& shape() const{
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


  template <char op2, class L2, class R2>
  const auto evaluate(const LazyFunction<op2, L2, R2>& f, std::size_t idx) const {
    return f(idx);
  }

  // TODO: collapse with previous.
  template <char op2, class L2, class R2, std::size_t dims>
  const auto evaluate(const LazyFunction<op2, L2, R2>& f, const std::array<std::size_t, dims>& idx) const {
    return f(idx);
  }


  template <class T> requires T::contiguous_storage
  const auto& evaluate(const T& x, std::size_t idx) const {
    return x[idx];
  }

  template <nd_object T, std::size_t dims>
  const auto& evaluate(const T& x, const std::array<std::size_t, dims>& idx) const {
    return x(idx);
  }

  template <class T, class Idx> requires std::is_scalar_v<T>
  const auto& evaluate(const T& x, const Idx& /*idx*/) const {
    return x;
  }

  const L& l_;
  const R& r_;
};

template<class T>
concept lazy_evaluated = nd_object<T> || std::is_scalar_v<T>;

template <lazy_evaluated L, lazy_evaluated R>
auto operator+(const L& l, const R& r) {
  return nd::LazyFunction<'+', L, R>(l, r);
}

template <lazy_evaluated L, lazy_evaluated R>
auto operator-(const L& l, const R& r) {
  return nd::LazyFunction<'-', L, R>(l, r);
}

template <lazy_evaluated L, lazy_evaluated R>
auto operator*(const L& l, const R& r) {
  return nd::LazyFunction<'*', L, R>(l, r);
}

template <lazy_evaluated L, lazy_evaluated R>
auto operator/(const L& l, const R& r) {
  return nd::LazyFunction<'/', L, R>(l, r);
}

}  // namespace nd
