//===-- Utility class to manipulate fixed point numbers. --*- C++ -*-=========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FIXEDPOINT_FXREP_H
#define LLVM_LIBC_SRC___SUPPORT_FIXEDPOINT_FXREP_H

#include "include/llvm-libc-macros/stdfix-macros.h"
#include "include/llvm-libc-types/stdfix_types.h"
#include "src/__support/macros/attributes.h"   // LIBC_INLINE
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

#include "fxrep.h"

#ifdef LIBC_COMPILER_HAS_FIXED_POINT

namespace LIBC_NAMESPACE::fixed_point {

template <typename T> LIBC_INLINE constexpr T abs(T x) {
  if constexpr (FXRep<T>::SIGN_LEN == 0)
    return x;
  else {
    if (LIBC_UNLIKELY(x == FXRep<T>::MIN()))
      return FXRep<T>::MAX();
    return (x < FXRep<T>::ZERO() ? -x : x);
  }
}

} // namespace LIBC_NAMESPACE::fixed_point

#endif // LIBC_COMPILER_HAS_FIXED_POINT

#endif // LLVM_LIBC_SRC___SUPPORT_FIXEDPOINT_FXREP_H
