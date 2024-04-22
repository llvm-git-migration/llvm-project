//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: c++20 || clang

#include <type_traits>
#include <utility>

struct trivially_copyable {
  int arr[4];
};

static_assert(std::is_trivially_copyable<std::pair<int, int>>::value, "");
static_assert(std::is_trivially_copyable<std::pair<int, char>>::value, "");
static_assert(std::is_trivially_copyable<std::pair<char, int>>::value, "");
static_assert(std::is_trivially_copyable<std::pair<std::pair<char, char>, int>>::value, "");
static_assert(std::is_trivially_copyable<std::pair<trivially_copyable, int>>::value, "");

static_assert(!std::is_trivially_copyable<std::pair<int&, int>>::value, "");
static_assert(!std::is_trivially_copyable<std::pair<int, int&>>::value, "");
static_assert(!std::is_trivially_copyable<std::pair<int&, int&>>::value, "");

static_assert(std::is_trivially_copy_constructible<std::pair<int, int>>::value);
static_assert(std::is_trivially_move_constructible<std::pair<int, int>>::value);
static_assert(std::is_trivially_copy_assignable<std::pair<int, int>>::value);
static_assert(std::is_trivially_move_assignable<std::pair<int, int>>::value);
static_assert(std::is_trivially_destructible<std::pair<int, int>>::value);
