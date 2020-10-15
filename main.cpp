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
}
