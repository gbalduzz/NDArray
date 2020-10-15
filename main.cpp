#include <array>
#include <iostream>
#include <vector>
#include <cassert>

template <class T, std::size_t dims>
class NDArray {
public:
  template <class... Ints>
  NDArray(Ints... ns) : shape_{std::size_t(ns)...} {
    static_assert(sizeof...(ns) == dims, "wrong number of parameters");
    static_assert((std::is_integral_v<Ints> && ...), "index must be integer");

    data_.resize((ns * ...), 0);

    strides_[0] = 1;
    for (int i = 1; i < dims; ++i)
      strides_[i] = strides_[i - 1] * shape_[i - 1];
  }

  std::size_t length() const noexcept {
    return data_.size();
  }

  const auto& shape() const noexcept {
    return shape_;
  }

  template <class... Ints>
  const T& operator()(Ints... ns) const noexcept {
    return data_[linindex(ns...)];
  }

  template <class... Ints>
  T& operator()(Ints... ns) noexcept {
    return data_[linindex(ns...)];
  }

private:
  template <class... Ints>
  std::size_t linindex(Ints... ids) const noexcept {
    static_assert(sizeof...(ids) == dims, "wrong number of parameters");
    static_assert((std::is_integral_v<Ints> && ...), "index must be integer");

    std::size_t lid = 0;
    unsigned i = 0;
    (..., (lid += ids * strides_[i++]));

    assert(((ids >= 0) &&...));
    assert((i = 0, ((ids < strides_[i++]) &&...)));

    return lid;
  }

  std::vector<T> data_;
  std::array<std::size_t, dims> shape_;
  std::array<std::size_t, dims> strides_;
};

int main() {
  NDArray<int, 2> arr(2, 3);
  arr(0, 0) = 1;
  std::cout << arr(0, 0) << std::endl;
}
