//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// [Example 1:
//   vector<string> vs = {"the", "quick", "brown", "fox"};
//   for (char c : vs | views::join_with('-')) {
//     cout << c;
//   }
//   // The above prints the-quick-brown-fox

#include <ranges>

#include <algorithm>
#include <cassert>
#include <string>
#include <string_view>
#include <vector>

using namespace std::string_view_literals;

constexpr bool test() {
  std::vector<std::string> vs = {"the", "quick", "brown", "fox"};
  std::string result;
  for (char c : vs | std::views::join_with('-')) {
    result += c;
  }

  return std::ranges::equal(result, "the-quick-brown-fox"sv);
}

int main(int, char**) {
  assert(test());
#if __cpp_lib_variant >= 202106 // TODO RANGES Remove when P2231R1 is implemented.
  static_assert(test());
#endif

  return 0;
}
