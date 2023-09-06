//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// template<class R, class P>
//   join_with_view(R&&, P&&) -> join_with_view<views::all_t<R>, views::all_t<P>>;
//
// template<input_range R>
//   join_with_view(R&&, range_value_t<range_reference_t<R>>)
//     -> join_with_view<views::all_t<R>, single_view<range_value_t<range_reference_t<R>>>>;

#include <ranges>

#include <cassert>
#include <deque>
#include <type_traits>

#include "test_iterators.h"

struct View : std::ranges::view_base {
  View() = default;
  cpp20_input_iterator<std::deque<int>*> begin() const;
  sentinel_wrapper<cpp20_input_iterator<std::deque<int>*>> end() const;
};
static_assert(std::ranges::view<View>);

struct Pattern : std::ranges::view_base {
  Pattern() = default;
  forward_iterator<int*> begin();
  forward_iterator<int*> end();
};
static_assert(std::ranges::view<Pattern>);

// A range that is not a view
struct Range {
  Range() = default;
  cpp20_input_iterator<std::deque<int>*> begin() const;
  sentinel_wrapper<cpp20_input_iterator<std::deque<int>*>> end() const;
};
static_assert(std::ranges::range<Range>);
static_assert(!std::ranges::view<Range>);

// A pattern that is not a view
struct RangePattern {
  RangePattern() = default;
  forward_iterator<int*> begin();
  forward_iterator<int*> end();
};
static_assert(std::ranges::range<RangePattern>);
static_assert(!std::ranges::view<RangePattern>);

constexpr void test_range_and_pattern_deduction_guide() {
  {
    View v;
    Pattern pat;
    std::ranges::join_with_view view(v, pat);
    static_assert(std::is_same_v<decltype(view), std::ranges::join_with_view<View, Pattern>>);
  }
  {
    Range r;
    Pattern pat;
    std::ranges::join_with_view view(r, pat);
    static_assert(std::is_same_v<decltype(view), std::ranges::join_with_view<std::ranges::ref_view<Range>, Pattern>>);
  }
  {
    View v;
    RangePattern pat;
    std::ranges::join_with_view view(v, pat);
    static_assert(
        std::is_same_v<decltype(view), std::ranges::join_with_view<View, std::ranges::ref_view<RangePattern>>>);
  }
  {
    Range v;
    RangePattern pat;
    std::ranges::join_with_view view(v, pat);
    static_assert(
        std::is_same_v<decltype(view),
                       std::ranges::join_with_view<std::ranges::ref_view<Range>, std::ranges::ref_view<RangePattern>>>);
  }
  {
    Pattern pat;
    std::ranges::join_with_view view(Range{}, pat);
    static_assert(
        std::is_same_v<decltype(view), std::ranges::join_with_view<std::ranges::owning_view<Range>, Pattern>>);
  }
  {
    View v;
    std::ranges::join_with_view view(v, RangePattern{});
    static_assert(
        std::is_same_v<decltype(view), std::ranges::join_with_view<View, std::ranges::owning_view<RangePattern>>>);
  }
  {
    std::ranges::join_with_view view(Range{}, RangePattern{});
    static_assert(
        std::is_same_v<
            decltype(view),
            std::ranges::join_with_view<std::ranges::owning_view<Range>, std::ranges::owning_view<RangePattern>>>);
  }
}

constexpr void test_range_and_element_deduction_guide() {
  {
    View v;
    std::ranges::join_with_view view(v, 0);
    static_assert(std::is_same_v<decltype(view), std::ranges::join_with_view<View, std::ranges::single_view<int>>>);
  }
  {
    Range r;
    std::ranges::join_with_view view(r, 1);
    static_assert(
        std::is_same_v<decltype(view),
                       std::ranges::join_with_view<std::ranges::ref_view<Range>, std::ranges::single_view<int>>>);
  }
  {
    std::ranges::join_with_view view(Range{}, 2);
    static_assert(
        std::is_same_v<decltype(view),
                       std::ranges::join_with_view<std::ranges::owning_view<Range>, std::ranges::single_view<int>>>);
  }
}

constexpr bool test() {
  test_range_and_pattern_deduction_guide();
  test_range_and_element_deduction_guide();

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
