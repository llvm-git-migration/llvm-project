//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

// void wait(T, memory_order = memory_order::seq_cst) const noexcept;

#include <atomic>
#include <cassert>
#include <type_traits>

#include "make_test_thread.h"
#include "test_macros.h"

template <typename T>
void test_wait() {
  T x(T(1));
  std::atomic_ref<T> a(x);

  assert(a.load() == T(1));
  a.wait(T(0));
  std::thread t1 = support::make_test_thread([&]() {
    a.store(T(3));
    a.notify_one();
  });
  a.wait(T(1));
  assert(a.load() == T(3));
  t1.join();
  ASSERT_NOEXCEPT(a.wait(T(0)));

  assert(a.load() == T(3));
  a.wait(T(0), std::memory_order_seq_cst);
  std::thread t2 = support::make_test_thread([&]() {
    a.store(T(5));
    a.notify_one();
  });
  a.wait(T(3), std::memory_order_seq_cst);
  assert(a.load() == T(5));
  t2.join();
  ASSERT_NOEXCEPT(a.wait(T(0), std::memory_order_seq_cst));
}

void test() {
  test_wait<int>();

  test_wait<float>();

  test_wait<int*>();

  struct X {
    int i;
    X(int ii) noexcept : i(ii) {}
    bool operator==(X o) const { return i == o.i; }
  };
  test_wait<X>();
}

int main(int, char**) {
  test();
  return 0;
}
