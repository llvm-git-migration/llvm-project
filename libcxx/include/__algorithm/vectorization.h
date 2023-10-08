//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_VECTORIZATION_H
#define _LIBCPP___ALGORITHM_VECTORIZATION_H

#include <__config>
#include <__type_traits/is_floating_point.h>
#include <__utility/integer_sequence.h>
#include <experimental/__simd/simd.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 17 && defined(_LIBCPP_ENABLE_EXPERIMENTAL)
#  define _LIBCPP_CAN_VECTORIZE_ALGORIHTMS 1
#else
#  define _LIBCPP_CAN_VECTORIZE_ALGORIHTMS 0
#endif

#if _LIBCPP_CAN_VECTORIZE_ALGORIHTMS && !defined(__OPTIMIZE_SIZE__)
#  define _LIBCPP_VECTORIZE_CLASSIC_ALGORITHMS 1
#else
#  define _LIBCPP_VECTORIZE_CLASSIC_ALGORITHMS 0
#endif

#if _LIBCPP_VECTORIZE_CLASSIC_ALGORITHMS && defined(__FAST_MATH__)
#  define _LIBCPP_VECTORIZE_FLOATING_POINT_CLASSIC_ALGORITHMS 1
#else
#  define _LIBCPP_VECTORIZE_FLOATING_POINT_CLASSIC_ALGORITHMS 0
#endif

#if _LIBCPP_CAN_VECTORIZE_ALGORIHTMS

_LIBCPP_BEGIN_NAMESPACE_STD

template <class _Tp>
inline static const bool __fits_in_vector =
    sizeof(_Tp) == 1 || sizeof(_Tp) == 2 || sizeof(_Tp) == 4 || sizeof(_Tp) == 8;

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI constexpr auto __get_arithmetic_type_impl() {
  if constexpr (is_floating_point_v<_Tp>)
    return _Tp{};
  else if constexpr (constexpr auto __sz = sizeof(_Tp); __sz == 1)
    return uint8_t{};
  else if constexpr (__sz == 2)
    return uint16_t{};
  else if constexpr (__sz == 4)
    return uint32_t{};
  else if constexpr (__sz == 8)
    return uint64_t{};
  else
    static_assert(false, "unexpected sizeof type");
}

template <class _Tp>
using __get_arithmetic_type = decltype(__get_arithmetic_type_impl<_Tp>());

template <class _Tp>
using __arithmetic_vec = experimental::native_simd<__get_arithmetic_type<_Tp>>;

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI __arithmetic_vec<_Tp> __load_as_arithmetic(_Tp* __values) {
  return {reinterpret_cast<__get_arithmetic_type<_Tp>*>(__values), 0};
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_CAN_VECTORIZE_ALGORIHTMS

#endif // _LIBCPP___ALGORITHM_VECTORIZATION_H
