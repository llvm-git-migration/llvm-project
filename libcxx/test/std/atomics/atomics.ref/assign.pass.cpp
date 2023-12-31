//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// T operator=(T) const noexcept;

#include <atomic>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

template <typename T>
void test_assign() {
  T x(T(1));
  std::atomic_ref<T> a(x);

  a = T(2);
  assert(x == T(2));

  ASSERT_NOEXCEPT(a = T(0));
  static_assert(std::is_nothrow_assignable_v<std::atomic_ref<T>, T>);

  static_assert(!std::is_copy_assignable_v<std::atomic_ref<T>>);
}

void test() {
  test_assign<int>();

  test_assign<float>();

  test_assign<int*>();

  struct X {
    int i;
    X(int ii) noexcept : i(ii) {}
    bool operator==(X o) const { return i == o.i; }
  };
  test_assign<X>();
}

int main(int, char**) {
  test();
  return 0;
}
