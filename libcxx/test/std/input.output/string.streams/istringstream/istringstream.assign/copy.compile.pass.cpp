//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <sstream>

// template <class charT, class traits = char_traits<charT>, class Allocator = allocator<charT> >
// class basic_stringstream

// basic_istringstream& operator=(const basic_istringstream&) = delete;

#include <sstream>
#include <type_traits>

static_assert(!std::is_copy_assignable<std::istringstream>::value,"");

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(!std::is_copy_assignable<std::wistringstream>::value,"");
#endif
