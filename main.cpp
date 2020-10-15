#include <iostream>

#include "nd_array.hpp"

template <class T>
std::ostream& operator<<(std::ostream& s, const std::vector<T>& v) {
  for (auto&& x : v)
    s << x << " ";
  return s << std::endl;
}
template <class T, std::size_t n>
std::ostream& operator<<(std::ostream& s, const std::array<T, n>& v) {
  for (auto&& x : v)
    s << x << " ";
  return s << std::endl;
}

int main() {
  using namespace nd;
  NDArray<int, 3> arr(5, 5, 5);
  auto arr2 = arr(range{2, end}, 2, all);

  std::cout << arr2.shape() << std::endl;

  NDArray<int, 2> m(2, 2);
  m = 1;

  NDArray<int, 2> m2(2, 2);
  m2 = 2;

  m(0, all) = m2(0, all);

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j)
      std::cout << m(i, j) << " ";
    std::cout << std::endl;
  }
}
