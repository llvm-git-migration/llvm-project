//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef _LIBCPP___FWD_TUPLE_H
#define _LIBCPP___FWD_TUPLE_H

#include <__config>
#include <__type_traits/remove_cvref.h>
#include <cstddef>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

template <size_t, class>
struct _LIBCPP_TEMPLATE_VIS tuple_element;

#ifndef _LIBCPP_CXX03_LANG

template <class...>
class _LIBCPP_TEMPLATE_VIS tuple;

template <class>
struct _LIBCPP_TEMPLATE_VIS tuple_size;

#  if _LIBCPP_STD_VER >= 20

template <class _Tp>
inline constexpr bool __is_tuple_like_v = false;

template <class... _Types>
inline constexpr bool __is_tuple_like_v<tuple<_Types...>> = true;

template <class _Tp>
concept __tuple_like = __is_tuple_like_v<remove_cvref_t<_Tp>>;

#  endif // _LIBCPP_STD_VER >= 20

template <size_t _Ip, class... _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 typename tuple_element<_Ip, tuple<_Tp...> >::type&
get(tuple<_Tp...>&) _NOEXCEPT;

template <size_t _Ip, class... _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 const typename tuple_element<_Ip, tuple<_Tp...> >::type&
get(const tuple<_Tp...>&) _NOEXCEPT;

template <size_t _Ip, class... _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 typename tuple_element<_Ip, tuple<_Tp...> >::type&&
get(tuple<_Tp...>&&) _NOEXCEPT;

template <size_t _Ip, class... _Tp>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR_SINCE_CXX14 const typename tuple_element<_Ip, tuple<_Tp...> >::type&&
get(const tuple<_Tp...>&&) _NOEXCEPT;

#endif // _LIBCPP_CXX03_LANG

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___FWD_TUPLE_H
