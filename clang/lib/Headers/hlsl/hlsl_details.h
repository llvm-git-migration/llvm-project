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
#define HLSL_BUILTIN_ATTRIBUTES\
  __attribute__((__always_inline__, __nodebug__)) static inline

template<bool B, typename T>
struct enable_if { };

template<typename T>
struct enable_if<true, T> {
  using Type = T;
};

template<bool B, typename T=void>
using enable_if_t = typename enable_if<B,T>::Type;

template<typename U, typename T, int N>
HLSL_BUILTIN_ATTRIBUTES vector<U, N> bit_cast(vector<T, N> V) {
  _Static_assert(sizeof(U) == sizeof(T), "casting types must be same bit size.");
  return __builtin_bit_cast(vector<U, N>, V);
}

template<typename U, typename T>
HLSL_BUILTIN_ATTRIBUTES U bit_cast(T F) {
  _Static_assert(sizeof(U) == sizeof(T), "casting types must be same bit size.");
  return __builtin_bit_cast(U, F);
}

} // namespace hlsl

#endif //_HLSL_HLSL_DETAILS_H_
