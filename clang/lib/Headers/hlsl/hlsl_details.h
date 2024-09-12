//===----- hlsl_intrinsics.h - HLSL definitions for intrinsics ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_HLSL_DETAILS_H_
#define _HLSL_HLSL_DETAILS_H_

namespace __details {
#define HLSL_INLINE_ATTRIBUTE                                                  \
  __attribute__((__always_inline__, __nodebug__)) static inline

template <bool B, typename T> struct enable_if {};

template <typename T> struct enable_if<true, T> { using Type = T; };

template <typename U, typename T, int N>
HLSL_INLINE_ATTRIBUTE
    typename enable_if<sizeof(U) == sizeof(T), vector<U, N> >::Type
    bit_cast(vector<T, N> V) {
  return __builtin_bit_cast(vector<U, N>, V);
}

template <typename U, typename T>
HLSL_INLINE_ATTRIBUTE typename enable_if<sizeof(U) == sizeof(T), U>::Type
bit_cast(T F) {
  return __builtin_bit_cast(U, F);
}

} // namespace __details

#endif //_HLSL_HLSL_DETAILS_H_
