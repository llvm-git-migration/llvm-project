//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <inplace_vector>

// static size_type capacity() noexcept;

#include <inplace_vector>
#include <concepts>
#include <cassert>

#include "test_macros.h"
#include "MoveOnly.h"

template <typename T, std::size_t N>
constexpr void test() {
  using V = std::inplace_vector<T, N>;
  V v;
  std::same_as<std::size_t> decltype(auto) sz = v.capacity();
  assert(sz == N);
  static_assert(v.capacity() == N);
  static_assert(V::capacity() == N);
  ASSERT_NOEXCEPT(v.capacity());
  ASSERT_NOEXCEPT(V::capacity());
}

constexpr bool tests() {
  test<int, 10>();
  test<int, 0>();
  test<MoveOnly, 0>();
  if
    !consteval { test<MoveOnly, 10>(); }
  {
    extern std::inplace_vector<MoveOnly, 10> v;
    assert(v.capacity() == 10);
  }

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
