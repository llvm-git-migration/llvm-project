//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___TUPLE_PAIR_LIKE_H
#define _LIBCPP___TUPLE_PAIR_LIKE_H

#include <__config>
#include <__fwd/array.h>
#include <__fwd/pair.h>
#include <__fwd/subrange.h>
#include <__fwd/tuple.h>
#include <__tuple/tuple_size.h>
#include <__type_traits/remove_cvref.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 20

template <class _Tp>
inline constexpr bool __is_tuple_like_v = false;

template <class _Tp, size_t _Size>
inline constexpr bool __is_tuple_like_v<array<_Tp, _Size>> = true;

template <class _T1, class _T2>
inline constexpr bool __is_tuple_like_v<pair<_T1, _T2>> = true;

template <class... _Types>
inline constexpr bool __is_tuple_like_v<tuple<_Types...>> = true;

template <class _Ip, class _Sp, ranges::subrange_kind _Kp>
inline constexpr bool __is_tuple_like_v<ranges::subrange<_Ip, _Sp, _Kp>> = true;

template <class _Tp>
concept __tuple_like = __is_tuple_like_v<remove_cvref_t<_Tp>>;

template <class _Tp>
concept __pair_like = __tuple_like<_Tp> && tuple_size<remove_cvref_t<_Tp>>::value == 2;

#endif // _LIBCPP_STD_VER >= 20

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___TUPLE_PAIR_LIKE_H
