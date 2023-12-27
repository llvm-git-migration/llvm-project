//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// TODO FMT This test should not require std::to_chars(floating-point)
// XFAIL: availability-fp_to_chars-missing

// <chrono>

// class system_clock;

// template<class charT, class traits, class Duration>
//   basic_ostream<charT, traits>&
//     operator<<(basic_ostream<charT, traits>& os, const sys_time<Duration>& tp);

// Constraints: treat_as_floating_point_v<typename Duration::rep> is false, and Duration{1} < days{1} is true.

#include <chrono>
#include <ratio>
#include <sstream>
#include <type_traits>

void test() {
  std::stringstream sstr;

  // floating-point values

  sstr << // expected-error {{invalid operands to binary expression}}
      std::chrono::sys_time<std::chrono::duration<float, std::ratio<1, 1>>>{
          std::chrono::duration<float, std::ratio<1, 1>>{0}};

  sstr << // expected-error {{invalid operands to binary expression}}
      std::chrono::sys_time<std::chrono::duration<double, std::ratio<1, 1>>>{
          std::chrono::duration<double, std::ratio<1, 1>>{0}};

  sstr << // expected-error {{invalid operands to binary expression}}
      std::chrono::sys_time<std::chrono::duration<long double, std::ratio<1, 1>>>{
          std::chrono::duration<long double, std::ratio<1, 1>>{0}};

  // duration >= day

  sstr << // valid since day has its own formatter
      std::chrono::sys_days{std::chrono::days{0}};

  using rep = std::conditional_t<std::is_same_v<std::chrono::days::rep, int>, long, int>;
  sstr << // a different rep does not matter,
      std::chrono::sys_time<std::chrono::duration<rep, std::ratio<86400>>>{
          std::chrono::duration<rep, std::ratio<86400>>{0}};

  sstr << // expected-error {{invalid operands to binary expression}}
      std::chrono::sys_time<std::chrono::duration<typename std::chrono::days::rep, std::ratio<86401>>>{
          std::chrono::duration<typename std::chrono::days::rep, std::ratio<86401>>{0}};

  sstr << // These are considered days.
      std::chrono::sys_time<std::chrono::weeks>{std::chrono::weeks{3}};

  sstr << // These too.
      std::chrono::sys_time<std::chrono::duration<rep, std::ratio<20 * 86400>>>{
          std::chrono::duration<rep, std::ratio<20 * 86400>>{0}};

  sstr << // expected-error {{invalid operands to binary expression}}
      std::chrono::sys_time<std::chrono::months>{std::chrono::months{0}};

  sstr << // expected-error {{invalid operands to binary expression}}
      std::chrono::sys_time<std::chrono::years>{std::chrono::years{0}};
}
