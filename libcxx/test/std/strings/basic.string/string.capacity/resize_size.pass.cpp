//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>

// void resize(size_type n); // constexpr since C++20

#include <string>
#include <stdexcept>
#include <cassert>

#include "test_macros.h"
#include "min_allocator.h"
#include "asan_testing.h"

template <class S>
TEST_CONSTEXPR_CXX20 void test(S s, typename S::size_type n, S expected) {
  s.resize(n);
  LIBCPP_ASSERT(s.__invariants());
  assert(s == expected);
  LIBCPP_ASSERT(is_string_asan_correct(s));
}

template <class S>
TEST_CONSTEXPR_CXX20 void test_string() {
  test(S(), 0, S());
  test(S(), 1, S(1, '\0'));
  test(S(), 10, S(10, '\0'));
  test(S(), 100, S(100, '\0'));
  test(S("12345"), 0, S());
  test(S("12345"), 2, S("12"));
  test(S("12345"), 5, S("12345"));
  test(S("12345"), 15, S("12345\0\0\0\0\0\0\0\0\0\0", 15));
  test(S("12345678901234567890123456789012345678901234567890"), 0, S());
  test(S("12345678901234567890123456789012345678901234567890"), 10, S("1234567890"));
  test(S("12345678901234567890123456789012345678901234567890"),
       50,
       S("12345678901234567890123456789012345678901234567890"));
  test(S("12345678901234567890123456789012345678901234567890"),
       60,
       S("12345678901234567890123456789012345678901234567890\0\0\0\0\0\0\0\0\0\0", 60));
}

template <class CharT>
TEST_CONSTEXPR_CXX20 void test_max_size() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  if (!TEST_IS_CONSTANT_EVALUATED) {
    std::basic_string<CharT> str;
    try {
      str.resize(std::string::npos);
      assert(false);
    } catch (const std::length_error&) {
    }
  }
#endif

  {
    std::basic_string<CharT, std::char_traits<CharT>, tiny_size_allocator<32, CharT>> str;
    str.resize(str.max_size());
    assert(str.size() == str.max_size());
  }
}

TEST_CONSTEXPR_CXX20 bool test() {
  test_string<std::string>();
#if TEST_STD_VER >= 11
  test_string<std::basic_string<char, std::char_traits<char>, min_allocator<char>>>();
  test_string<std::basic_string<char, std::char_traits<char>, safe_allocator<char>>>();
#endif

  test_max_size<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_max_size<wchar_t>();
#endif

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
