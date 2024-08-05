//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// UNSUPPORTED: c++03, c++11, c++17

#include <tuple>

[[maybe_unused]] std::tuple_size<std::tuple<void, void>> test;

// expected-warning@*:* {{'__volatile_deprecated_since_cxx20_warning<volatile std::tuple<void, void>>' is deprecated}}
[[maybe_unused]] std::tuple_size<volatile std::tuple<void, void>> vol_test;

// expected-warning@*:* {{'__volatile_deprecated_since_cxx20_warning<const volatile std::tuple<void, void>>' is deprecated}}
[[maybe_unused]] std::tuple_size<const volatile std::tuple<void, void>> const_vol_test;
