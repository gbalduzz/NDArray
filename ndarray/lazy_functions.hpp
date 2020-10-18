#pragma once

#include <vector>
#include <array>

namespace nd {

template <char op, class L, class R>
class LazyFunction {
public:
  LazyFunction(const L& l, const R& r) : l_(l), r_(r) {}

  auto operator()(const std::size_t idx) const {
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

private:
  template <char op2, class L2, class R2>
  const auto evaluate(const LazyFunction<op2, L2, R2>& f, std::size_t idx) const {
    return f(idx);
  }

  template <class T> requires T::contiguous_storage
  const auto& evaluate(const T& x, std::size_t idx) const {
    return x[idx];
  }

  template <class T>
  requires std::is_scalar_v<T> const auto& evaluate(const T& x, std::size_t /*idx*/) const {
    return x;
  }

  const L& l_;
  const R& r_;
};

template <class L, class R>
auto operator+(const L& l, const R& r) {
  return nd::LazyFunction<'+', L, R>(l, r);
}

template <class L, class R>
auto operator-(const L& l, const R& r) {
  return nd::LazyFunction<'-', L, R>(l, r);
}

template <class L, class R>
auto operator*(const L& l, const R& r) {
  return nd::LazyFunction<'*', L, R>(l, r);
}

template <class L, class R>
auto operator/(const L& l, const R& r) {
  return nd::LazyFunction<'/', L, R>(l, r);
}

}  // namespace nd
