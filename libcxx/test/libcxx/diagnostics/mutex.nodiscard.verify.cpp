//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// check that <mutex> functions are marked [[nodiscard]]

#include <mutex>

#include "test_macros.h"

void test() {
  std::mutex mutex;
  std::lock_guard{mutex}; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
  std::lock_guard{mutex, std::adopt_lock}; // expected-warning {{ignoring return value of function declared with 'nodiscard' attribute}}
}
