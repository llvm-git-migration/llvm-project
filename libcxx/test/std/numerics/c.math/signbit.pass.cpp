//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bool signbit(floating-point-type x); // constexpr since C++23

// We don't control the implementation on windows
// UNSUPPORTED: windows

#include <cassert>
#include <cmath>
#include <limits>

#include "test_macros.h"
#include "type_algorithms.h"

struct TestFloat {
  template <class T>
  static TEST_CONSTEXPR_CXX23 bool test() {
    assert(!std::signbit(T(0)));
    assert(std::signbit(-T(0)));
    assert(std::signbit(std::numeric_limits<T>::lowest()));
    assert(!std::signbit(std::numeric_limits<T>::min()));
    assert(!std::signbit(std::numeric_limits<T>::denorm_min()));
    assert(!std::signbit(std::numeric_limits<T>::max()));
    assert(!std::signbit(std::numeric_limits<T>::infinity()));
    assert(std::signbit(-std::numeric_limits<T>::infinity()));
    assert(!std::signbit(std::numeric_limits<T>::quiet_NaN()));
    assert(!std::signbit(std::numeric_limits<T>::signaling_NaN()));

    return true;
  }

  template <class T>
  TEST_CONSTEXPR_CXX23 void operator()() {
    test<T>();
#if TEST_STD_VER >= 23
    static_assert(test<T>());
#endif
  }
};

struct TestUnsignedIntAndFixedWidthChar {
  template <class T>
  static TEST_CONSTEXPR_CXX23 bool test() {
    assert(!std::signbit(std::numeric_limits<T>::max()));
    assert(!std::signbit(std::numeric_limits<T>::lowest()));
    assert(!std::signbit(T(0)));

    return true;
  }

  template <class T>
  TEST_CONSTEXPR_CXX23 void operator()() {
    test<T>();
#if TEST_STD_VER >= 23
    static_assert(test<T>());
#endif
  }
};

struct TestSignedIntAndVariableWidthChar {
  template <class T>
  static TEST_CONSTEXPR_CXX23 bool test() {
    assert(!std::signbit(std::numeric_limits<T>::max()));
    assert(std::signbit(std::numeric_limits<T>::lowest()));
    assert(!std::signbit(T(0)));

    return true;
  }

// Plain `char` on PowerPC and ARM defaults to be a `unsigned char` (contrary to
// e.g. x86) and therefore `std::lowest()` returns 0.
#if defined(__arm__) && defined(__powerpc__)
  template <>
  TEST_CONSTEXPR_CXX23 bool test<char>() {
    assert(!std::signbit(std::numeric_limits<char>::max()));
    assert(!std::signbit(std::numeric_limits<char>::lowest()));
    assert(!std::signbit(char(0)));

    return true;
  }
#endif

  template <class T>
  TEST_CONSTEXPR_CXX23 void operator()() {
    test<T>();
#if TEST_STD_VER >= 23
    static_assert(test<T>());
#endif
  }
};

template <typename T>
struct ConvertibleTo {
  operator T() const { return T(); }
};

int main(int, char**) {
  types::for_each(types::floating_point_types(), TestFloat());
  types::for_each(types::concatenate_t<types::unsigned_integer_types, types::fixed_width_character_types>(),
                  TestUnsignedIntAndFixedWidthChar());
  types::for_each(types::concatenate_t<types::signed_integer_types, types::variable_width_character_types>(),
                  TestSignedIntAndVariableWidthChar());

  // Make sure we can call `std::signbit` with convertible types. This checks
  // whether overloads for all cv-unqualified floating-point types are working
  // as expected.
  {
    assert(!std::signbit(ConvertibleTo<float>()));
    assert(!std::signbit(ConvertibleTo<double>()));
    assert(!std::signbit(ConvertibleTo<long double>()));
  }

  return 0;
}
