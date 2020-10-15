#pragma once

#include <cassert>
#include <type_traits>

namespace nd {

struct range { // [start, end)
  std::size_t start;
  std::size_t end;
};

constexpr std::size_t end = -1;
constexpr range all{0, end};

template <std::size_t N, class... Args>
constexpr bool is_complete_index = sizeof...(Args) == N && (std::is_integral_v<Args> && ...);

template <std::size_t N, class... Args>
constexpr bool is_partial_index = !is_complete_index<N, Args...> &&
                                  sizeof...(Args) <= N &&
                                  ((std::is_integral_v<Args> || std::is_same_v<Args, range>)&&...);

template <std::size_t N, class... Args>
constexpr std::size_t free_dimensions = N - (std::size_t(std::is_integral_v<Args>) +...);

std::size_t getStart(const range& r) {
  return r.start;
}

template <std::integral I>
std::size_t getStart(I i) {
  return i;
}

std::size_t getSpan(const range& r, const std::size_t tot_size) {
  assert(r.end == end || (r.end >= r.start && r.end <= tot_size));
  return r.end == end? tot_size - r.start : r.end - r.start;
}

template <std::integral I>
std::size_t getSpan(const I idx, const std::size_t tot_size) {
  return 0;
}

}  // namespace nd
