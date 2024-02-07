//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <functional>

// template<class F, class... Args>
//   constexpr unspecified bind_back(F&& f, Args&&... args);

#include <functional>

#include "types.h"

void f() {
  int n       = 1;
  const int c = 1;

  auto p = std::bind_back(pass, c);
  static_assert(p() == 1); // expected-error {{static assertion expression is not an integral constant expression}}

  auto d = std::bind_back(do_nothing, n); // expected-error {{no matching function for call to 'bind_back'}}

  auto t = std::bind_back( // expected-error {{no matching function for call to 'bind_back'}}
      testNotMoveConst,
      NotMoveConst(0));
}
