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
#include "src/__support/macros/attributes.h" // LIBC_INLINE, LIBC_INLINE_VAR

#ifdef LIBC_COMPILER_HAS_FIXED_POINT

namespace LIBC_NAMESPACE::fixedpoint {

template <typename T> FXRep;

template <> FXRep<short fract> {
  using Type = short fract;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = 0;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = SFRACT_FBIT;
  LIBC_INLINE static constexpr Type MIN() { return SFRACT_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return SFRACT_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0HR; }
  LIBC_INLINE static constexpr Type EPS() { return SFRACT_EPSILON; }
}

template <> FXRep<unsigned short fract> {
  using Type = unsigned short fract;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 0;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = 0;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = USFRACT_FBIT;
  LIBC_INLINE static constexpr Type MIN() { return USFRACT_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return USFRACT_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0UHR; }
  LIBC_INLINE static constexpr Type EPS() { return USFRACT_EPSILON; }
}

template <> FXRep<fract> {
  using Type = fract;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = 0;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = FRACT_FBIT;
  LIBC_INLINE static constexpr Type MIN() { return FRACT_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return FRACT_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0R; }
  LIBC_INLINE static constexpr Type EPS() { return FRACT_EPSILON; }
}

template <> FXRep<unsigned fract> {
  using Type = unsigned fract;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 0;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = 0;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = UFRACT_FBIT;
  LIBC_INLINE static constexpr Type MIN() { return UFRACT_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return UFRACT_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0UR; }
  LIBC_INLINE static constexpr Type EPS() { return UFRACT_EPSILON; }
}

template <> FXRep<long fract> {
  using Type = long fract;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = 0;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = LFRACT_FBIT;
  LIBC_INLINE static constexpr Type MIN() { return LFRACT_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return LFRACT_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0LR; }
  LIBC_INLINE static constexpr Type EPS() { return LFRACT_EPSILON; }
}

template <> FXRep<unsigned long fract> {
  using Type = unsigned long fract;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 0;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = 0;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = ULFRACT_FBIT;
  LIBC_INLINE static constexpr Type MIN() { return ULFRACT_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return ULFRACT_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0ULR; }
  LIBC_INLINE static constexpr Type EPS() { return ULFRACT_EPSILON; }
}

template <> FXRep<short accum> {
  using Type = short accum;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = SACCUM_IBIT;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = SACCUM_FBIT;
  LIBC_INLINE static constexpr Type MIN() { return SACCUM_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return SACCUM_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0HK; }
  LIBC_INLINE static constexpr Type EPS() { return SACCUM_EPSILON; }
}

template <> FXRep<unsigned short accum> {
  using Type = unsigned short accum;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 0;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = UACCUM_IBIT;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = USACCUM_FBIT;
  LIBC_INLINE static constexpr Type MIN() { return USACCUM_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return USACCUM_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0UHK; }
  LIBC_INLINE static constexpr Type EPS() { return USACCUM_EPSILON; }
}

template <> FXRep<accum> {
  using Type = accum;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = ACCUM_IBIT;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = ACCUM_FBIT;
  LIBC_INLINE static constexpr Type MIN() { return ACCUM_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return ACCUM_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0K; }
  LIBC_INLINE static constexpr Type EPS() { return ACCUM_EPSILON; }
}

template <> FXRep<unsigned accum> {
  using Type = unsigned accum;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 0;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = UACCUM_IBIT;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = UACCUM_FBIT;
  LIBC_INLINE static constexpr Type MIN() { return UACCUM_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return UACCUM_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0UK; }
  LIBC_INLINE static constexpr Type EPS() { return UACCUM_EPSILON; }
}

template <> FXRep<long accum> {
  using Type = long accum;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 1;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = LACCUM_IBIT;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = LACCUM_FBIT;
  LIBC_INLINE static constexpr Type MIN() { return LACCUM_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return LACCUM_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0LK; }
  LIBC_INLINE static constexpr Type EPS() { return LACCUM_EPSILON; }
}

template <> FXRep<unsigned long accum> {
  using Type = unsigned long accum;
  LIBC_INLINE_VAR static constexpr int SIGN_LEN = 0;
  LIBC_INLINE_VAR static constexpr int INTEGRAL_LEN = ULACCUM_IBIT;
  LIBC_INLINE_VAR static constexpr int FRACTION_LEN = ULACCUM_FBIT;
  LIBC_INLINE static constexpr Type MIN() { return ULACCUM_MIN; }
  LIBC_INLINE static constexpr Type MAX() { return ULACCUM_MIN; }
  LIBC_INLINE static constexpr Type ZERO() { return 0ULK; }
  LIBC_INLINE static constexpr Type EPS() { return ULACCUM_EPSILON; }
}

} // namespace LIBC_NAMESPACE::fixedpoint

#endif // LIBC_COMPILER_HAS_FIXED_POINT

#endif // LLVM_LIBC_SRC___SUPPORT_FIXEDPOINT_FXREP_H
