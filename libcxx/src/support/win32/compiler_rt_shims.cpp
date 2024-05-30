//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//
// This file reimplements builtins that are normally provided by compiler-rt, which is
// not provided on Windows. This should go away once compiler-rt is shipped on Windows.
//

#include <cmath>

extern "C" _LIBCPP_EXPORTED_FROM_ABI _Complex double __muldc3(double __a, double __b, double __c, double __d) {
  double __ac = __a * __c;
  double __bd = __b * __d;
  double __ad = __a * __d;
  double __bc = __b * __c;
  _Complex double z;
  if (std::isnan(__real__ z) && std::isnan(__imag__ z)) {
    int __recalc = 0;
    if (std::isinf(__a) || std::isinf(__b)) {
      __a = std::copysign(std::isinf(__a) ? 1 : 0, __a);
      __b = std::copysign(std::isinf(__b) ? 1 : 0, __b);
      if (std::isnan(__c))
        __c = std::copysign(0, __c);
      if (std::isnan(__d))
        __d = std::copysign(0, __d);
      __recalc = 1;
    }
    if (std::isinf(__c) || std::isinf(__d)) {
      __c = std::copysign(std::isinf(__c) ? 1 : 0, __c);
      __d = std::copysign(std::isinf(__d) ? 1 : 0, __d);
      if (std::isnan(__a))
        __a = std::copysign(0, __a);
      if (std::isnan(__b))
        __b = std::copysign(0, __b);
      __recalc = 1;
    }
    if (!__recalc && (std::isinf(__ac) || std::isinf(__bd) || std::isinf(__ad) || std::isinf(__bc))) {
      if (std::isnan(__a))
        __a = std::copysign(0, __a);
      if (std::isnan(__b))
        __b = std::copysign(0, __b);
      if (std::isnan(__c))
        __c = std::copysign(0, __c);
      if (std::isnan(__d))
        __d = std::copysign(0, __d);
      __recalc = 1;
    }
    if (__recalc) {
      __real__ z = HUGE_VAL * (__a * __c - __b * __d);
      __imag__ z = HUGE_VAL * (__a * __d + __b * __c);
    }
  }
  return z;
}

extern "C" _LIBCPP_EXPORTED_FROM_ABI _Complex float __mulsc3(float __a, float __b, float __c, float __d) {
  float __ac = __a * __c;
  float __bd = __b * __d;
  float __ad = __a * __d;
  float __bc = __b * __c;
  _Complex float z;
  __real__ z = __ac - __bd;
  __imag__ z = __ad + __bc;
  if (std::isnan(__real__ z) && std::isnan(__imag__ z)) {
    int __recalc = 0;
    if (std::isinf(__a) || std::isinf(__b)) {
      __a = std::copysignf(std::isinf(__a) ? 1 : 0, __a);
      __b = std::copysignf(std::isinf(__b) ? 1 : 0, __b);
      if (std::isnan(__c))
        __c = std::copysignf(0, __c);
      if (std::isnan(__d))
        __d = std::copysignf(0, __d);
      __recalc = 1;
    }
    if (std::isinf(__c) || std::isinf(__d)) {
      __c = std::copysignf(std::isinf(__c) ? 1 : 0, __c);
      __d = std::copysignf(std::isinf(__d) ? 1 : 0, __d);
      if (std::isnan(__a))
        __a = std::copysignf(0, __a);
      if (std::isnan(__b))
        __b = std::copysignf(0, __b);
      __recalc = 1;
    }
    if (!__recalc && (std::isinf(__ac) || std::isinf(__bd) || std::isinf(__ad) || std::isinf(__bc))) {
      if (std::isnan(__a))
        __a = std::copysignf(0, __a);
      if (std::isnan(__b))
        __b = std::copysignf(0, __b);
      if (std::isnan(__c))
        __c = std::copysignf(0, __c);
      if (std::isnan(__d))
        __d = std::copysignf(0, __d);
      __recalc = 1;
    }
    if (__recalc) {
      __real__ z = HUGE_VALF * (__a * __c - __b * __d);
      __imag__ z = HUGE_VALF * (__a * __d + __b * __c);
    }
  }
  return z;
}

extern "C" _LIBCPP_EXPORTED_FROM_ABI _Complex double __divdc3(double __a, double __b, double __c, double __d) {
  int __ilogbw   = 0;
  double __logbw = std::logb(std::fmax(std::fabs(__c), std::fabs(__d)));
  if (std::isfinite(__logbw)) {
    __ilogbw = (int)__logbw;
    __c      = std::scalbn(__c, -__ilogbw);
    __d      = std::scalbn(__d, -__ilogbw);
  }
  double __denom = __c * __c + __d * __d;
  _Complex double z;
  __real__ z = std::scalbn((__a * __c + __b * __d) / __denom, -__ilogbw);
  __imag__ z = std::scalbn((__b * __c - __a * __d) / __denom, -__ilogbw);
  if (std::isnan(__real__ z) && std::isnan(__imag__ z)) {
    if ((__denom == 0.0) && (!std::isnan(__a) || !std::isnan(__b))) {
      __real__ z = std::copysign(HUGE_VAL, __c) * __a;
      __imag__ z = std::copysign(HUGE_VAL, __c) * __b;
    } else if ((std::isinf(__a) || std::isinf(__b)) && std::isfinite(__c) && std::isfinite(__d)) {
      __a        = std::copysign(std::isinf(__a) ? 1.0 : 0.0, __a);
      __b        = std::copysign(std::isinf(__b) ? 1.0 : 0.0, __b);
      __real__ z = HUGE_VAL * (__a * __c + __b * __d);
      __imag__ z = HUGE_VAL * (__b * __c - __a * __d);
    } else if (std::isinf(__logbw) && __logbw > 0.0 && std::isfinite(__a) && std::isfinite(__b)) {
      __c        = std::copysign(std::isinf(__c) ? 1.0 : 0.0, __c);
      __d        = std::copysign(std::isinf(__d) ? 1.0 : 0.0, __d);
      __real__ z = 0.0 * (__a * __c + __b * __d);
      __imag__ z = 0.0 * (__b * __c - __a * __d);
    }
  }
  return z;
}

extern "C" _LIBCPP_EXPORTED_FROM_ABI _Complex float __divsc3(float __a, float __b, float __c, float __d) {
  int __ilogbw  = 0;
  float __logbw = std::logbf(__builtin_fmaxf(std::fabsf(__c), std::fabsf(__d)));
  if (std::isfinite(__logbw)) {
    __ilogbw = (int)__logbw;
    __c      = std::scalbnf(__c, -__ilogbw);
    __d      = std::scalbnf(__d, -__ilogbw);
  }
  float __denom = __c * __c + __d * __d;
  _Complex float z;
  __real__ z = std::scalbnf((__a * __c + __b * __d) / __denom, -__ilogbw);
  __imag__ z = std::scalbnf((__b * __c - __a * __d) / __denom, -__ilogbw);
  if (std::isnan(__real__ z) && std::isnan(__imag__ z)) {
    if ((__denom == 0) && (!std::isnan(__a) || !std::isnan(__b))) {
      __real__ z = std::copysignf(HUGE_VALF, __c) * __a;
      __imag__ z = std::copysignf(HUGE_VALF, __c) * __b;
    } else if ((std::isinf(__a) || std::isinf(__b)) && std::isfinite(__c) && std::isfinite(__d)) {
      __a        = std::copysignf(std::isinf(__a) ? 1 : 0, __a);
      __b        = std::copysignf(std::isinf(__b) ? 1 : 0, __b);
      __real__ z = HUGE_VALF * (__a * __c + __b * __d);
      __imag__ z = HUGE_VALF * (__b * __c - __a * __d);
    } else if (std::isinf(__logbw) && __logbw > 0 && std::isfinite(__a) && std::isfinite(__b)) {
      __c        = std::copysignf(std::isinf(__c) ? 1 : 0, __c);
      __d        = std::copysignf(std::isinf(__d) ? 1 : 0, __d);
      __real__ z = 0 * (__a * __c + __b * __d);
      __imag__ z = 0 * (__b * __c - __a * __d);
    }
  }
  return z;
}
