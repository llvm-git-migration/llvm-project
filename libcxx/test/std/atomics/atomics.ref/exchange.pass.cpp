//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// T exchange(T, memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <type_traits>

#include "test_macros.h"

template <typename T>
void test_exchange() {
  T x(T(1));
  std::atomic_ref<T> a(x);

  assert(a.exchange(T(2)) == T(1));
  ASSERT_NOEXCEPT(a.exchange(T(2)));

  assert(a.exchange(T(3), std::memory_order_seq_cst) == T(2));
  ASSERT_NOEXCEPT(a.exchange(T(3), std::memory_order_seq_cst));
}

void test() {
  test_exchange<int>();

  test_exchange<float>();

  test_exchange<int*>();

  struct X {
    int i;
    X(int ii) noexcept : i(ii) {}
    bool operator==(X o) const { return i == o.i; }
  };
  test_exchange<X>();
}

int main(int, char**) {
  test();
  return 0;
}
