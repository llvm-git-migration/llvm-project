//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// join_with_view()
//   requires default_initializable<V> && default_initializable<Pattern> = default;

#include <ranges>

#include <algorithm>
#include <array>
#include <cassert>
#include <concepts>
#include <type_traits>

using InnerRange = std::array<int, 2>;
using OuterRange = std::array<InnerRange, 3>;

struct TrivialView : std::ranges::view_base {
  int val_; // intentionally uninitialised

  TrivialView() = default;
  const InnerRange* begin();
  const InnerRange* end();
};

static_assert(std::is_trivial_v<TrivialView>);

struct DefaultView : std::ranges::view_base {
  static constexpr OuterRange view = {{{1, 2}, {3, 4}, {5, 6}}};
  const OuterRange* r_;

  constexpr DefaultView(const OuterRange* r = &view) : r_(r) {}

  constexpr const InnerRange* begin() { return r_->data(); }
  constexpr const InnerRange* end() { return r_->data() + r_->size(); }
};

struct NonDefaultConstructibleView : TrivialView {
  NonDefaultConstructibleView(int);
};

struct Pattern : std::ranges::view_base {
  int val; // intentionally uninitialised

  constexpr int* begin() { return &val; }
  constexpr int* end() { return &val + 1; }
};

struct NonDefaultConstructiblePattern : Pattern {
  NonDefaultConstructiblePattern(int);
};

constexpr bool test() {
  // Check if `base_` is value initialised
  {
    std::ranges::join_with_view<TrivialView, Pattern> v;
    assert(std::move(v).base().val_ == 0);
  }

  // Check if `pattern_` is value initialised
  {
    std::ranges::join_with_view<DefaultView, Pattern> v;
    assert(std::ranges::equal(v, std::array{1, 2, 0, 3, 4, 0, 5, 6}));
    assert(std::move(v).base().r_ == &DefaultView::view);
  }

  static_assert(std::default_initializable<std::ranges::join_with_view<TrivialView, Pattern>>);
  static_assert(!std::default_initializable<std::ranges::join_with_view<TrivialView, NonDefaultConstructiblePattern>>);
  static_assert(!std::default_initializable<std::ranges::join_with_view<NonDefaultConstructibleView, Pattern>>);
  static_assert(!std::default_initializable<
                std::ranges::join_with_view<NonDefaultConstructibleView, NonDefaultConstructiblePattern>>);

  return true;
}

int main() {
  test();
#if __cpp_lib_variant >= 202106 // TODO RANGES Remove when P2231R1 is implemented.
  static_assert(test());
#endif

  return 0;
}
