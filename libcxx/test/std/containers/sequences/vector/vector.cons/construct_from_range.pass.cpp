//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<container-compatible-range<T> R>
//   vector(from_range_t, R&& rg, const Allocator& = Allocator()); // C++23

#include <cassert>
#include <sstream>
#include <vector>

#include "../../from_range_sequence_containers.h"
#include "asan_testing.h"
#include "test_macros.h"

constexpr bool test() {
  for_all_iterators_and_allocators<int>([]<class Iter, class Sent, class Alloc>() {
    test_sequence_container<std::vector, int, Iter, Sent, Alloc>([]([[maybe_unused]] const auto& c) {
      LIBCPP_ASSERT(c.__invariants());
      LIBCPP_ASSERT(is_contiguous_container_asan_correct(c));
    });
  });
  test_sequence_container_move_only<std::vector>();

  return true;
}

void test_sized_input_only_range() {
  std::istringstream is{"1 2 3 4"};
  auto vals = std::views::istream<int>(is);
  std::vector v(std::from_range, std::views::counted(vals.begin(), 3));
  assert(v == (std::vector{1, 2, 3}));
}

int main(int, char**) {
  static_assert(test_constraints<std::vector, int, double>());
  test();

  static_assert(test());

  test_exception_safety_throwing_copy<std::vector>();
  test_exception_safety_throwing_allocator<std::vector, int>();

  test_sized_input_only_range();

  return 0;
}
