//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// struct __private_constructor_tag {};

// The private constructor tag is intended to be a trivial type that can easily
// be used to mark a constructor exposition-only. The name is uglified and not
// provided directly by the utility header.
//
// Tests whether the type is not provided by the utility header.

#include <utility>

// expected-error@+1 {{no type named '__private_constructor_tag' in namespace 'std'}}
std::__private_constructor_tag tag;
