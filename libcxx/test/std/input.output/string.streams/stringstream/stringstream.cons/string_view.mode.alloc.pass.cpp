//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT>>
// class basic_stringstream

// template<class T>
//   basic_stringstream(const T& t, ios_base::openmode which, const Allocator& a);

#include <cassert>
#include <concepts>
#include <sstream>
#include <string>
#include <string_view>

#include "make_string.h"
#include "test_allocator.h"
#include "test_convertible.h"
#include "test_macros.h"

template <typename CharT>
void test_sfinae() {
  struct SomeObject {};
  struct NonAllocator {};

  // `const CharT*`
  static_assert(std::constructible_from<std::basic_stringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                        const CharT*,
                                        test_allocator<CharT>>);
  static_assert(test_convertible<std::basic_stringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                 const CharT*,
                                 std::ios_base::openmode,
                                 const test_allocator<CharT>>());
  // `std::basic_string_view<CharT>`
  static_assert(std::constructible_from<std::basic_stringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                        std::basic_string_view<CharT>,
                                        std::ios_base::openmode,
                                        test_allocator<CharT>>);
  static_assert(test_convertible<std::basic_stringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                 const std::basic_string_view<CharT>,
                                 std::ios_base::openmode,
                                 const test_allocator<CharT>>());
  // `std::basic_string<CharT>`
  static_assert(std::constructible_from<std::basic_stringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                        const std::basic_string<CharT>,
                                        std::ios_base::openmode,
                                        const test_allocator<CharT>>);
  static_assert(test_convertible<std::basic_stringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                 const std::basic_string<CharT>,
                                 std::ios_base::openmode,
                                 const test_allocator<CharT>>());

  // String-view-like
  static_assert(!std::constructible_from<std::basic_stringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                         const int,
                                         std::ios_base::openmode,
                                         const test_allocator<CharT>>);
  static_assert(!test_convertible<std::basic_stringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  const int,
                                  std::ios_base::openmode,
                                  const test_allocator<CharT>>());

  static_assert(!std::constructible_from<std::basic_stringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                         const SomeObject,
                                         std::ios_base::openmode,
                                         const test_allocator<CharT>>);
  static_assert(!test_convertible<std::basic_stringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  const SomeObject,
                                  std::ios_base::openmode,
                                  const test_allocator<CharT>>());

  // Mode
  static_assert(!std::constructible_from<std::basic_stringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                         const std::basic_string_view<CharT>,
                                         SomeObject,
                                         const NonAllocator>);
  static_assert(!test_convertible<std::basic_stringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  const std::basic_string_view<CharT>,
                                  SomeObject,
                                  const NonAllocator>());

  // Allocator
  static_assert(!std::constructible_from<std::basic_stringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                         const std::basic_string_view<CharT>,
                                         std::ios_base::openmode,
                                         const NonAllocator>);
  static_assert(!test_convertible<std::basic_stringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>>,
                                  const std::basic_string_view<CharT>,
                                  std::ios_base::openmode,
                                  const NonAllocator>());
}

#define CS(S) MAKE_CSTRING(CharT, S)
#define ST(S) MAKE_STRING(CharT, S)
#define SV(S) MAKE_STRING_VIEW(CharT, S)
#include <print>

template <class CharT>
static void test() {
  const test_allocator<CharT> ca;

  // const CharT*
  {
    std::basic_stringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(
        CS("zmt"), std::ios_base::out | std::ios_base::in, ca);
    assert(ss.str() == CS("zmt"));
    assert(ss.rdbuf()->get_allocator() == ca);
  }
  // std::basic_string_view<CharT>
  {
    const auto csv = SV("zmt");
    std::basic_stringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(
        csv, std::ios_base::out | std::ios_base::in, ca);
    assert(ss.str() == CS("zmt"));
    assert(ss.rdbuf()->get_allocator() == ca);
  }
  // std::basic_string<CharT>
  {
    const auto cs = ST("zmt");
    std::basic_stringstream<CharT, std::char_traits<CharT>, test_allocator<CharT>> ss(
        cs, std::ios_base::out | std::ios_base::in, ca);
    assert(ss.str() == CS("zmt"));
    assert(ss.rdbuf()->get_allocator() == ca);
  }
}

int main(int, char**) {
  test_sfinae<char>();
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_sfinae<wchar_t>();
  test<wchar_t>();
#endif
  return 0;
}
