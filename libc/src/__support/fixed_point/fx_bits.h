//===-- Utility class to manipulate fixed point numbers. --*- C++ -*-=========//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_FIXEDPOINT_FXBITS_H
#define LLVM_LIBC_SRC___SUPPORT_FIXEDPOINT_FXBITS_H

#include "include/llvm-libc-macros/stdfix-macros.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/CPP/type_traits.h"
#include "src/__support/macros/attributes.h"   // LIBC_INLINE
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY

#include "fx_rep.h"

#ifdef LIBC_COMPILER_HAS_FIXED_POINT

namespace LIBC_NAMESPACE::fixed_point {

template <typename T> struct FXBits {
private:
  using fx_rep = FXRep<T>;
  using StorageType = typename fx_rep::StorageType;

  StorageType value;

  static_assert(fx_rep::FRACTION_LEN > 0);

  static constexpr size_t FRACTION_OFFSET = 0; // Just for completeness
  static constexpr size_t INTEGRAL_OFFSET = fx_rep::FRACTION_LEN;
  static constexpr size_t SIGN_OFFSET =
      fx_rep::SIGN_LEN == 0
          ? 0
          : ((sizeof(StorageType) * CHAR_BIT) - fx_rep::SIGN_LEN);

  static constexpr StorageType FRACTION_MASK =
      (StorageType(1) << fx_rep::FRACTION_LEN) - 1;
  static constexpr StorageType INTEGRAL_MASK =
      ((StorageType(1) << fx_rep::INTEGRAL_LEN) - 1) << INTEGRAL_OFFSET;
  static constexpr StorageType SIGN_MASK =
      (fx_rep::SIGN_LEN == 0 ? 0 : StorageType(1) << SIGN_OFFSET);

public:
  LIBC_INLINE constexpr FXBits() = default;

  template <typename XType> LIBC_INLINE constexpr explicit FXBits(XType x) {
    using Unqual = typename cpp::remove_cv_t<XType>;
    if constexpr (cpp::is_same_v<Unqual, T>) {
      value = cpp::bit_cast<StorageType>(x);
    } else if constexpr (cpp::is_same_v<Unqual, StorageType>) {
      value = x;
    } else {
      // We don't want accidental type promotions/conversions, so we require
      // exact type match.
      static_assert(cpp::always_false<XType>);
    }
  }

  LIBC_INLINE constexpr StorageType get_fraction() {
    return (value & FRACTION_MASK) >> FRACTION_OFFSET;
  }

  LIBC_INLINE constexpr StorageType get_integral() {
    return (value & INTEGRAL_MASK) >> INTEGRAL_OFFSET;
  }

  LIBC_INLINE constexpr StorageType get_sign() {
    return (value & SIGN_MASK) >> SIGN_OFFSET;
  }

  LIBC_INLINE constexpr void set_fraction(StorageType fraction) {
    value = (value & (~FRACTION_MASK)) |
            ((fraction << FRACTION_OFFSET) & FRACTION_MASK);
  }

  LIBC_INLINE constexpr void set_integral(StorageType integral) {
    value = (value & (~INTEGRAL_MASK)) |
            ((integral << INTEGRAL_OFFSET) & INTEGRAL_MASK);
  }

  LIBC_INLINE constexpr void set_sign(StorageType sign) {
    value = (value & (~SIGN_MASK)) | ((sign << SIGN_OFFSET) & SIGN_MASK);
  }

  LIBC_INLINE constexpr T get_val() const { return cpp::bit_cast<T>(value); }
};

template <typename T> LIBC_INLINE constexpr T abs(T x) {
  using FXRep = FXRep<T>;
  if constexpr (FXRep::SIGN_LEN == 0)
    return x;
  else {
    if (LIBC_UNLIKELY(x == FXRep::MIN()))
      return FXRep::MAX();
    return (x < FXRep::ZERO() ? -x : x);
  }
}

} // namespace LIBC_NAMESPACE::fixed_point

#endif // LIBC_COMPILER_HAS_FIXED_POINT

#endif // LLVM_LIBC_SRC___SUPPORT_FIXEDPOINT_FXBITS_H
