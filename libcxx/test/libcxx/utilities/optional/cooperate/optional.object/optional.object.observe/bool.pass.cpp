//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// <optional>

// constexpr explicit optional<T>::operator bool() const noexcept;

#include <optional>
#include <functional>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

using std::optional;

template<typename T, typename F>
constexpr bool test(F init) {
  {
    const optional<T> opt; (void) opt;
    ASSERT_NOEXCEPT(bool(opt));
    static_assert(!std::is_convertible_v<optional<T>, bool>);
  }
  {
    constexpr optional<T> opt;
    static_assert(!opt);
  }
  {
    constexpr optional<T> opt(init());
    static_assert(opt);
  }
  return true;
}

int f() { return 0; }

int main(int, char**) {
  static int i;
  test<std::reference_wrapper<int>>([]() -> auto& { return i; });
  test<std::reference_wrapper<int()>>([]() -> auto& { return f; });
}
