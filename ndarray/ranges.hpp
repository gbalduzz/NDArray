#pragma once

#include <cassert>
#include <type_traits>

namespace nd {

struct range {  // [start, end)
  long int start;
  long int end;
};

constexpr long int end = 0;
constexpr range all{0, end};

template <std::size_t N, class... Args>
constexpr bool is_complete_index = sizeof...(Args) == N &&
                                   (std::is_integral_v<std::decay_t<Args>> && ...);

template <std::size_t N, class... Args>
constexpr bool is_partial_index =
    !is_complete_index<N, Args...> && sizeof...(Args) <= N &&
    ((std::is_integral_v<std::decay_t<Args>> || std::is_same_v<std::decay_t<Args>, range>)&&...);

template <std::size_t N, class... Args>
constexpr std::size_t free_dimensions = N -
                                        (std::size_t(std::is_integral_v<std::decay_t<Args>>) + ...);

template <std::integral I>
std::size_t getStart(I i, const std::size_t shape) {
  return i >= 0 ? i : shape + i;
}

std::size_t getStart(const range& r, const std::size_t shape) {
  return getStart(r.start, shape);
}

std::size_t getSpan(const range& r, const std::size_t shape) {
  const std::size_t idx_end = r.end > 0 ? r.end : shape + r.end;
  assert(idx_end <= shape && idx_end > r.start);
  return idx_end - r.start;
}

template <std::integral I>
std::size_t getSpan(const I idx, const std::size_t shape) {
  return 0;
}

}  // namespace nd
