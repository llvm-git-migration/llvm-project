//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: __builtin_popcountg is available since Clang 19 and GCC 14. When support for older versions is dropped, we can
//  refactor this code to exclusively use __builtin_popcountg.

#ifndef _LIBCPP___BIT_POPCOUNT_H
#define _LIBCPP___BIT_POPCOUNT_H

#include <__bit/rotate.h>
#include <__concepts/arithmetic.h>
#include <__config>
#include <__type_traits/enable_if.h>
#include <__type_traits/is_unsigned.h>
#include <limits>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR int __libcpp_popcount(unsigned __x) _NOEXCEPT {
  return __builtin_popcount(__x);
}

inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR int __libcpp_popcount(unsigned long __x) _NOEXCEPT {
  return __builtin_popcountl(__x);
}

inline _LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR int __libcpp_popcount(unsigned long long __x) _NOEXCEPT {
  return __builtin_popcountll(__x);
}

#if _LIBCPP_STD_VER >= 17
// constexpr implementation for C++17 and later
template <class _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr int __popcount(_Tp __t) _NOEXCEPT {
  static_assert(is_unsigned<_Tp>::value, "__popcount only works with unsigned types");
  if constexpr (sizeof(_Tp) <= sizeof(unsigned int)) {
    return std::__libcpp_popcount(static_cast<unsigned int>(__t));
  } else if constexpr (sizeof(_Tp) <= sizeof(unsigned long)) {
    return std::__libcpp_popcount(static_cast<unsigned long>(__t));
  } else if constexpr (sizeof(_Tp) <= sizeof(unsigned long long)) {
    return std::__libcpp_popcount(static_cast<unsigned long long>(__t));
  } else {
    int __ret = 0;
    while (__t != 0) {
      __ret += std::__libcpp_popcount(static_cast<unsigned long long>(__t));
      __t >>= std::numeric_limits<unsigned long long>::digits;
    }
    return __ret;
  }
}

#else
// constexpr implementation for C++11 and C++14

template < class _Tp, __enable_if_t<is_unsigned<_Tp>::value && sizeof(_Tp) <= sizeof(unsigned int), int> = 0>
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR int __popcount(_Tp __t) {
  return std::__libcpp_popcount(static_cast<unsigned int>(__t));
}

template < class _Tp,
           __enable_if_t<is_unsigned<_Tp>::value && (sizeof(_Tp) > sizeof(unsigned int)) &&
                             sizeof(_Tp) <= sizeof(unsigned long),
                         int> = 0 >
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR int __popcount(_Tp __t) {
  return std::__libcpp_popcount(static_cast<unsigned long>(__t));
}

template < class _Tp,
           __enable_if_t<is_unsigned<_Tp>::value && (sizeof(_Tp) > sizeof(unsigned long)) &&
                             sizeof(_Tp) <= sizeof(unsigned long long),
                         int> = 0 >
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR int __popcount(_Tp __t) {
  return std::__libcpp_popcount(static_cast<unsigned long long>(__t));
}

template < class _Tp, __enable_if_t<is_unsigned<_Tp>::value && (sizeof(_Tp) > sizeof(unsigned long long)), int> = 0 >
_LIBCPP_HIDE_FROM_ABI _LIBCPP_CONSTEXPR int __popcount(_Tp __t) {
#  if _LIBCPP_STD_VER == 11
  // A recursive constexpr implementation for C++11
  return __t != 0 ? std::__libcpp_popcount(static_cast<unsigned long long>(__t)) +
                        std::__popcount<_Tp>(__t >> numeric_limits<unsigned long long>::digits)
                  : 0;
#  else
  int __ret = 0;
  while (__t != 0) {
    __ret += std::__libcpp_popcount(static_cast<unsigned long long>(__t));
    __t >>= std::numeric_limits<unsigned long long>::digits;
  }
  return __ret;
}
#  endif // _LIBCPP_STD_VER == 11
}

#endif // _LIBCPP_CXX03_LANG

#if _LIBCPP_STD_VER >= 20

template <__libcpp_unsigned_integer _Tp>
[[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr int popcount(_Tp __t) noexcept {
#  if __has_builtin(__builtin_popcountg)
  return __builtin_popcountg(__t);
#  else
  return std::__popcount(__t);
#  endif
}

#endif

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS

#endif // _LIBCPP___BIT_POPCOUNT_H
