//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14

// <mutex>

// template <class ...Mutex> class scoped_lock;

// Test that we properly apply [[nodiscard]] to scoped_lock's constructors.

#include <mutex>

void f() {
  using M = std::mutex;
  M m0, m1, m2;
  // clang-format off
  std::scoped_lock<>{}; // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
  std::scoped_lock<M>{m0}; // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
  std::scoped_lock<M, M>{m0, m1}; // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
  std::scoped_lock<M, M, M>{m0, m1, m2}; // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}

  std::scoped_lock<>{std::adopt_lock}; // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
  std::scoped_lock<M>{std::adopt_lock, m0}; // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
  std::scoped_lock<M, M>{std::adopt_lock, m0, m1}; // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
  std::scoped_lock<M, M, M>{std::adopt_lock, m0, m1, m2}; // expected-warning {{ignoring temporary created by a constructor declared with 'nodiscard' attribute}}
  // clang-format on
}
