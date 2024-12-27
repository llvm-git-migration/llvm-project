//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17

// template<class T, output_iterator<const T&> O>
//   constexpr O ranges::fill_n(O first, iter_difference_t<O> n, const T& value);

#include <algorithm>
#include <array>
#include <cassert>
#include <ranges>
#include <string>
#include <vector>

#include "almost_satisfies_types.h"
#include "test_iterators.h"

template <class Iter>
concept HasFillN = requires(Iter iter) { std::ranges::fill_n(iter, int{}, int{}); };

struct WrongType {};

static_assert(HasFillN<int*>);
static_assert(!HasFillN<WrongType*>);
static_assert(!HasFillN<OutputIteratorNotIndirectlyWritable>);
static_assert(!HasFillN<OutputIteratorNotInputOrOutputIterator>);

template <class It, class Sent = It>
constexpr void test_iterators() {
  { // simple test
    int a[3];
    std::same_as<It> decltype(auto) ret = std::ranges::fill_n(It(a), 3, 1);
    assert(std::all_of(a, a + 3, [](int i) { return i == 1; }));
    assert(base(ret) == a + 3);
  }

  { // check that an empty range works
    std::array<int, 0> a;
    auto ret = std::ranges::fill_n(It(a.data()), 0, 1);
    assert(base(ret) == a.data());
  }
}

template <std::size_t N>
struct TestBitIter {
  constexpr void operator()() {
    { // Test fill_n with full bytes
      std::vector<bool> in(N);
      std::vector<bool> expected(N, true);
      std::ranges::fill_n(std::ranges::begin(in), N, true);
      assert(in == expected);
    }
    { // Test fill_n with partial bytes
      std::vector<bool> in(N + 4);
      std::vector<bool> expected(N + 4, true);
      std::ranges::fill_n(std::ranges::begin(in), N + 4, true);
      assert(in == expected);
    }
  }
};

constexpr bool test() {
  test_iterators<cpp17_output_iterator<int*>, sentinel_wrapper<cpp17_output_iterator<int*>>>();
  test_iterators<cpp20_output_iterator<int*>, sentinel_wrapper<cpp20_output_iterator<int*>>>();
  test_iterators<forward_iterator<int*>>();
  test_iterators<bidirectional_iterator<int*>>();
  test_iterators<random_access_iterator<int*>>();
  test_iterators<contiguous_iterator<int*>>();
  test_iterators<int*>();

  { // check that every element is copied once
    struct S {
      bool copied = false;
      constexpr S& operator=(const S&) {
        assert(!copied);
        copied = true;
        return *this;
      }
    };

    S a[5];
    std::ranges::fill_n(a, 5, S{});
    assert(std::all_of(a, a + 5, [](S& s) { return s.copied; }));
  }

  { // check that non-trivially copyable items are copied properly
    std::array<std::string, 10> a;
    auto ret = std::ranges::fill_n(a.data(), 10, "long long string so no SSO");
    assert(ret == a.data() + a.size());
    assert(std::all_of(a.begin(), a.end(), [](auto& s) { return s == "long long string so no SSO"; }));
  }

  { // Test vector<bool>::iterator optimization
    TestBitIter<8>()();
    TestBitIter<16>()();
    TestBitIter<32>()();
    TestBitIter<64>()();
    TestBitIter<256>()();
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
