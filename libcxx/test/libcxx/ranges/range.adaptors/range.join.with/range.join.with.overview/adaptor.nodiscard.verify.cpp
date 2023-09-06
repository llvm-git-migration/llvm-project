//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// Test the libc++ extension that std::views::join_with is marked as [[nodiscard]].

#include <ranges>

void test() {
  int range[][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
  int pattern_base[] = {-1, -1};
  auto pattern = std::views::all(pattern_base);

  std::views::join_with(pattern); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::join_with(range, pattern); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  range | std::views::join_with(pattern); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::all | std::views::join_with(pattern); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}

  std::views::join_with(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::join_with(range, 0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  range | std::views::join_with(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::views::all | std::views::join_with(0); // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
