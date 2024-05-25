//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// template<class F>
// function(F) -> function<see-below>;

// UNSUPPORTED: c++03, c++11, c++14

// The deduction guides for std::function do not handle rvalue-ref qualified
// call operators and C-style variadics. It also doesn't deduce from nullptr_t.
// Make sure we stick to the specification.

#include <functional>

struct R { };
struct f0 { R operator()() && { return {}; } };
struct f1 { R operator()(int, ...) { return {}; } };

void f() {
    std::function f = f0{}; // expected-error-re{{no viable constructor or deduction guide for deduction of template arguments of '{{(std::)?}}function'}}
    std::function g = f1{}; // expected-error-re{{no viable constructor or deduction guide for deduction of template arguments of '{{(std::)?}}function'}}
    std::function h = nullptr; // expected-error-re{{no viable constructor or deduction guide for deduction of template arguments of '{{(std::)?}}function'}}
}
