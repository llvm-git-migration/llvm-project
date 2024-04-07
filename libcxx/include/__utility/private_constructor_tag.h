// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP__UTILITY_PRIVATE_CONSTRUCTOR_TAG_H
#define _LIBCPP__UTILITY_PRIVATE_CONSTRUCTOR_TAG_H

#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// This struct is intended to use to make a non-standard constructor
// exposition-only. This does not prevent users from using the constructor, but
// they need to explicitly use __ugly_code and include a private header.
struct __private_constructor_tag {};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP__UTILITY_PRIVATE_CONSTRUCTOR_TAG_H
