//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TYPE_TRAITS_IS_TRIVIALLY_RELOCATABLE_H
#define _LIBCPP___TYPE_TRAITS_IS_TRIVIALLY_RELOCATABLE_H

#include <__config>
#include <__type_traits/enable_if.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_trivially_copyable.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

// A type is trivially relocatable if a move construct + destroy of the original object is equivalent to
// `memcpy(dst, src, sizeof(T))`.
//
// Note that we don't use Clang's __is_trivially_relocatable builtin because it doesn't honor the presence
// of non-trivial special members like assignment operators, or even a copy constructor, making it possible
// to incorrectly optimize operations that should call user-provided operations instead.
template <class _Tp, class = void>
struct __libcpp_is_trivially_relocatable : is_trivially_copyable<_Tp> {};

template <class _Tp>
struct __libcpp_is_trivially_relocatable<_Tp,
                                         __enable_if_t<is_same<_Tp, typename _Tp::__trivially_relocatable>::value> >
    : true_type {};

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TYPE_TRAITS_IS_TRIVIALLY_RELOCATABLE_H
