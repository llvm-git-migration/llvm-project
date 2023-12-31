//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// <atomic>

// explicit atomic_ref(T&);

#include <atomic>
#include <type_traits>

#include "test_macros.h"

template <typename T>
auto makeAtomicRef(T& obj) {
  // check that the constructor is explicit
  static_assert(!std::is_convertible_v<T, std::atomic_ref<T>>);
  static_assert(std::is_constructible_v<std::atomic_ref<T>, T&>);
  return std::atomic_ref<T>(obj);
}

void test() {
  int i = 0;
  (void)makeAtomicRef(i);

  float f = 0.f;
  (void)makeAtomicRef(f);

  int* p = &i;
  (void)makeAtomicRef(p);

  struct X {
  } x;
  (void)makeAtomicRef(x);
}

int main(int, char**) {
  test();
  return 0;
}
