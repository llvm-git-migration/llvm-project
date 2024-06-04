//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

#include "test_macros.h"

TEST_DIAGNOSTIC_PUSH
TEST_CLANG_DIAGNOSTIC_IGNORED("-Wprivate-header")
#include <__utility/pointer_int_pair.h>
TEST_DIAGNOSTIC_POP

// expected-error-re@*:* {{{{static_assert|static assertion}} failed {{.*}}Not enough bits available for requested bit count}}
std::__pointer_int_pair<char*, size_t, std::__integer_width{2}> ptr1; // expected-note {{here}}
// expected-error-re@*:* {{{{static_assert|static assertion}} failed {{.*}}_IntType has to be an integral type}}
std::__pointer_int_pair<int*, int*, std::__integer_width{2}> ptr2; // expected-note {{here}}
// expected-error-re@*:* {{{{static_assert|static assertion}} failed {{.*}}__pointer_int_pair doesn't work for signed types}}
std::__pointer_int_pair<int*, signed, std::__integer_width{2}> ptr3; // expected-note {{here}}
