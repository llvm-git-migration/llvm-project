//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// constexpr explicit join_with_view(V base, Pattern pattern);

#include <ranges>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <type_traits>
#include <utility>

#include "helpers.h"

struct View : std::ranges::view_base {
  using InnerRange = std::array<int, 2>;
  using OuterRange = std::array<InnerRange, 3>;

  static constexpr OuterRange default_range = {{{1, 2}, {3, 4}, {5, 6}}};
  static constexpr OuterRange range_on_move = {{{6, 5}, {4, 3}, {2, 1}}};

  constexpr View() : r_(&default_range) {}
  constexpr View(const View&) : r_(&default_range) {}
  constexpr View(View&&) : r_(&range_on_move) {}

  constexpr View& operator=(const View&) {
    r_ = &default_range;
    return *this;
  }

  constexpr View& operator=(View&&) {
    r_ = &default_range;
    return *this;
  }

  const InnerRange* begin() { return r_->data(); }
  const InnerRange* end() { return r_->data() + r_->size(); }

private:
  const OuterRange* r_;
};

struct Pattern : std::ranges::view_base {
  using PatternRange = std::array<int, 2>;

  static constexpr PatternRange default_range = {0, 0};
  static constexpr PatternRange range_on_move = {7, 7};

  constexpr Pattern() : r_(&default_range) {}
  constexpr Pattern(const Pattern&) : r_(&default_range) {}
  constexpr Pattern(Pattern&&) : r_(&range_on_move) {}

  constexpr Pattern& operator=(const Pattern&) {
    r_ = &default_range;
    return *this;
  }

  constexpr Pattern& operator=(Pattern&&) {
    r_ = &default_range;
    return *this;
  }

  const int* begin() { return r_->data(); }
  const int* end() { return r_->data() + r_->size(); }

private:
  const PatternRange* r_;
};

constexpr bool test() {
  // Check construction with `view` and `pattern` (glvalues)
  {
    View v;
    Pattern p;
    std::ranges::join_with_view<View, Pattern> jw{v, p};
    assert(std::ranges::equal(jw, std::array{6, 5, 7, 7, 4, 3, 7, 7, 2, 1}));
  }

  // Check construction with `view` and `pattern` (const glvalues)
  {
    const View v;
    const Pattern p;
    std::ranges::join_with_view<View, Pattern> jw{v, p};
    assert(std::ranges::equal(jw, std::array{6, 5, 7, 7, 4, 3, 7, 7, 2, 1}));
  }

  // Check construction with `view` and `pattern` (prvalues)
  {
    std::ranges::join_with_view<View, Pattern> jw{View{}, Pattern{}};
    assert(std::ranges::equal(jw, std::array{6, 5, 7, 7, 4, 3, 7, 7, 2, 1}));
  }

  // Check construction with `view` and `pattern` (xvalues)
  {
    View v;
    Pattern p;
    std::ranges::join_with_view<View, Pattern> jw{std::move(v), std::move(p)};
    assert(std::ranges::equal(jw, std::array{6, 5, 7, 7, 4, 3, 7, 7, 2, 1}));
  }

  // Check explicitness
  static_assert(ConstructionIsExplicit<std::ranges::join_with_view<View, Pattern>, View&, Pattern&>);
  static_assert(ConstructionIsExplicit<std::ranges::join_with_view<View, Pattern>, const View&, const Pattern&>);
  static_assert(ConstructionIsExplicit<std::ranges::join_with_view<View, Pattern>, View, Pattern>);
  static_assert(ConstructionIsExplicit<std::ranges::join_with_view<View, Pattern>, View&&, Pattern&&>);

  return true;
}

int main() {
  test();
#if __cpp_lib_variant >= 202106 // TODO RANGES Remove when P2231R1 is implemented.
  static_assert(test());
#endif

  return 0;
}
