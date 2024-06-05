//===- MPInt.h - MPInt Class ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a simple class to represent arbitrary precision signed integers.
// Unlike APInt, one does not have to specify a fixed maximum size, and the
// integer can take on any arbitrary values. This is optimized for small-values
// by providing fast-paths for the cases when the value stored fits in 64-bits.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_MPINT_H
#define LLVM_ADT_MPINT_H

#include "llvm/ADT/SlowMPInt.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

namespace llvm {
namespace detail {
/// ---------------------------------------------------------------------------
/// Some helpers from MLIR/MathExtras.
/// ---------------------------------------------------------------------------
LLVM_ATTRIBUTE_ALWAYS_INLINE int64_t ceilDiv(int64_t Numerator,
                                             int64_t Denominator) {
  assert(Denominator);
  if (!Numerator)
    return 0;
  // C's integer division rounds towards 0.
  int64_t X = (Denominator > 0) ? -1 : 1;
  bool SameSign = (Numerator > 0) == (Denominator > 0);
  return SameSign ? ((Numerator + X) / Denominator) + 1
                  : -(-Numerator / Denominator);
}

LLVM_ATTRIBUTE_ALWAYS_INLINE int64_t floorDiv(int64_t Numerator,
                                              int64_t Denominator) {
  assert(Denominator);
  if (!Numerator)
    return 0;
  // C's integer division rounds towards 0.
  int64_t X = (Denominator > 0) ? -1 : 1;
  bool SameSign = (Numerator > 0) == (Denominator > 0);
  return SameSign ? Numerator / Denominator
                  : -((-Numerator + X) / Denominator) - 1;
}

/// Returns the remainder of the Euclidean division of LHS by RHS. Result is
/// always non-negative.
LLVM_ATTRIBUTE_ALWAYS_INLINE int64_t mod(int64_t Numerator,
                                         int64_t Denominator) {
  assert(Denominator >= 1);
  return Numerator % Denominator < 0 ? Numerator % Denominator + Denominator
                                     : Numerator % Denominator;
}

/// If builtin intrinsics for overflow-checked arithmetic are available,
/// use them. Otherwise, call through to LLVM's overflow-checked arithmetic
/// functionality. Those functions also have such macro-gated uses of intrinsics
/// but they are not always_inlined, which is important for us to achieve
/// high-performance; calling the functions directly would result in a slowdown
/// of 1.15x.
LLVM_ATTRIBUTE_ALWAYS_INLINE bool addOverflow(int64_t X, int64_t Y,
                                              int64_t &Result) {
#if __has_builtin(__builtin_add_overflow)
  return __builtin_add_overflow(X, Y, &Result);
#else
  return AddOverflow(x, y, result);
#endif
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool subOverflow(int64_t X, int64_t Y,
                                              int64_t &Result) {
#if __has_builtin(__builtin_sub_overflow)
  return __builtin_sub_overflow(X, Y, &Result);
#else
  return SubOverflow(x, y, result);
#endif
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool mulOverflow(int64_t X, int64_t Y,
                                              int64_t &Result) {
#if __has_builtin(__builtin_mul_overflow)
  return __builtin_mul_overflow(X, Y, &Result);
#else
  return MulOverflow(x, y, result);
#endif
}
} // namespace detail

/// This class provides support for multi-precision arithmetic.
///
/// Unlike APInt, this extends the precision as necessary to prevent overflows
/// and supports operations between objects with differing internal precisions.
///
/// This is optimized for small-values by providing fast-paths for the cases
/// when the value stored fits in 64-bits. We annotate all fastpaths by using
/// the LLVM_LIKELY/LLVM_UNLIKELY annotations. Removing these would result in
/// a 1.2x performance slowdown.
///
/// We always_inline all operations; removing these results in a 1.5x
/// performance slowdown.
///
/// When holdsLarge is true, a SlowMPInt is held in the union. If it is false,
/// the int64_t is held. Using std::variant instead would lead to significantly
/// worse performance.
class MPInt {
private:
  union {
    int64_t ValSmall;
    detail::SlowMPInt ValLarge;
  };
  unsigned HoldsLarge;

  LLVM_ATTRIBUTE_ALWAYS_INLINE void initSmall(int64_t O) {
    if (LLVM_UNLIKELY(isLarge()))
      ValLarge.detail::SlowMPInt::~SlowMPInt();
    ValSmall = O;
    HoldsLarge = false;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE void initLarge(const detail::SlowMPInt &O) {
    if (LLVM_LIKELY(isSmall())) {
      // The data in memory could be in an arbitrary state, not necessarily
      // corresponding to any valid state of ValLarge; we cannot call any member
      // functions, e.g. the assignment operator on it, as they may access the
      // invalid internal state. We instead construct a new object using
      // placement new.
      new (&ValLarge) detail::SlowMPInt(O);
    } else {
      // In this case, we need to use the assignment operator, because if we use
      // placement-new as above we would lose track of allocated memory
      // and leak it.
      ValLarge = O;
    }
    HoldsLarge = true;
  }

  LLVM_ATTRIBUTE_ALWAYS_INLINE explicit MPInt(const detail::SlowMPInt &Val)
      : ValLarge(Val), HoldsLarge(true) {}
  LLVM_ATTRIBUTE_ALWAYS_INLINE bool isSmall() const { return !HoldsLarge; }
  LLVM_ATTRIBUTE_ALWAYS_INLINE bool isLarge() const { return HoldsLarge; }
  /// Get the stored value. For getSmall/Large,
  /// the stored value should be small/large.
  LLVM_ATTRIBUTE_ALWAYS_INLINE int64_t getSmall() const {
    assert(isSmall() &&
           "getSmall should only be called when the value stored is small!");
    return ValSmall;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE int64_t &getSmall() {
    assert(isSmall() &&
           "getSmall should only be called when the value stored is small!");
    return ValSmall;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE const detail::SlowMPInt &getLarge() const {
    assert(isLarge() &&
           "getLarge should only be called when the value stored is large!");
    return ValLarge;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE detail::SlowMPInt &getLarge() {
    assert(isLarge() &&
           "getLarge should only be called when the value stored is large!");
    return ValLarge;
  }
  explicit operator detail::SlowMPInt() const {
    if (isSmall())
      return detail::SlowMPInt(getSmall());
    return getLarge();
  }

public:
  LLVM_ATTRIBUTE_ALWAYS_INLINE explicit MPInt(int64_t Val)
      : ValSmall(Val), HoldsLarge(false) {}
  LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt() : MPInt(0) {}
  LLVM_ATTRIBUTE_ALWAYS_INLINE ~MPInt() {
    if (LLVM_UNLIKELY(isLarge()))
      ValLarge.detail::SlowMPInt::~SlowMPInt();
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt(const MPInt &O)
      : ValSmall(O.ValSmall), HoldsLarge(false) {
    if (LLVM_UNLIKELY(O.isLarge()))
      initLarge(O.ValLarge);
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt &operator=(const MPInt &O) {
    if (LLVM_LIKELY(O.isSmall())) {
      initSmall(O.ValSmall);
      return *this;
    }
    initLarge(O.ValLarge);
    return *this;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt &operator=(int X) {
    initSmall(X);
    return *this;
  }
  LLVM_ATTRIBUTE_ALWAYS_INLINE explicit operator int64_t() const {
    if (isSmall())
      return getSmall();
    return static_cast<int64_t>(getLarge());
  }

  bool operator==(const MPInt &O) const;
  bool operator!=(const MPInt &O) const;
  bool operator>(const MPInt &O) const;
  bool operator<(const MPInt &O) const;
  bool operator<=(const MPInt &O) const;
  bool operator>=(const MPInt &O) const;
  MPInt operator+(const MPInt &O) const;
  MPInt operator-(const MPInt &O) const;
  MPInt operator*(const MPInt &O) const;
  MPInt operator/(const MPInt &O) const;
  MPInt operator%(const MPInt &O) const;
  MPInt &operator+=(const MPInt &O);
  MPInt &operator-=(const MPInt &O);
  MPInt &operator*=(const MPInt &O);
  MPInt &operator/=(const MPInt &O);
  MPInt &operator%=(const MPInt &O);
  MPInt operator-() const;
  MPInt &operator++();
  MPInt &operator--();

  // Divide by a number that is known to be positive.
  // This is slightly more efficient because it saves an overflow check.
  MPInt divByPositive(const MPInt &O) const;
  MPInt &divByPositiveInPlace(const MPInt &O);

  friend MPInt abs(const MPInt &X);
  friend MPInt ceilDiv(const MPInt &LHS, const MPInt &RHS);
  friend MPInt floorDiv(const MPInt &LHS, const MPInt &RHS);
  // The operands must be non-negative for gcd.
  friend MPInt gcd(const MPInt &A, const MPInt &B);
  friend MPInt lcm(const MPInt &A, const MPInt &B);
  friend MPInt mod(const MPInt &LHS, const MPInt &RHS);

  /// ---------------------------------------------------------------------------
  /// Convenience operator overloads for int64_t.
  /// ---------------------------------------------------------------------------
  friend MPInt &operator+=(MPInt &A, int64_t B);
  friend MPInt &operator-=(MPInt &A, int64_t B);
  friend MPInt &operator*=(MPInt &A, int64_t B);
  friend MPInt &operator/=(MPInt &A, int64_t B);
  friend MPInt &operator%=(MPInt &A, int64_t B);

  friend bool operator==(const MPInt &A, int64_t B);
  friend bool operator!=(const MPInt &A, int64_t B);
  friend bool operator>(const MPInt &A, int64_t B);
  friend bool operator<(const MPInt &A, int64_t B);
  friend bool operator<=(const MPInt &A, int64_t B);
  friend bool operator>=(const MPInt &A, int64_t B);
  friend MPInt operator+(const MPInt &A, int64_t B);
  friend MPInt operator-(const MPInt &A, int64_t B);
  friend MPInt operator*(const MPInt &A, int64_t B);
  friend MPInt operator/(const MPInt &A, int64_t B);
  friend MPInt operator%(const MPInt &A, int64_t B);

  friend bool operator==(int64_t A, const MPInt &B);
  friend bool operator!=(int64_t A, const MPInt &B);
  friend bool operator>(int64_t A, const MPInt &B);
  friend bool operator<(int64_t A, const MPInt &B);
  friend bool operator<=(int64_t A, const MPInt &B);
  friend bool operator>=(int64_t A, const MPInt &B);
  friend MPInt operator+(int64_t A, const MPInt &B);
  friend MPInt operator-(int64_t A, const MPInt &B);
  friend MPInt operator*(int64_t A, const MPInt &B);
  friend MPInt operator/(int64_t A, const MPInt &B);
  friend MPInt operator%(int64_t A, const MPInt &B);

  friend hash_code hash_value(const MPInt &x); // NOLINT

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  raw_ostream &print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif
};

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
inline raw_ostream &operator<<(raw_ostream &OS, const MPInt &X) {
  X.print(OS);
  return OS;
}
#endif

/// Redeclarations of friend declaration above to
/// make it discoverable by lookups.
hash_code hash_value(const MPInt &X); // NOLINT

/// This just calls through to the operator int64_t, but it's useful when a
/// function pointer is required. (Although this is marked inline, it is still
/// possible to obtain and use a function pointer to this.)
static inline int64_t int64FromMPInt(const MPInt &X) { return int64_t(X); }
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt mpintFromInt64(int64_t X) {
  return MPInt(X);
}

// The RHS is always expected to be positive, and the result
/// is always non-negative.
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt mod(const MPInt &LHS, const MPInt &RHS);

namespace detail {
// Division overflows only when trying to negate the minimal signed value.
LLVM_ATTRIBUTE_ALWAYS_INLINE bool divWouldOverflow(int64_t X, int64_t Y) {
  return X == std::numeric_limits<int64_t>::min() && Y == -1;
}
} // namespace detail

/// We define the operations here in the header to facilitate inlining.

/// ---------------------------------------------------------------------------
/// Comparison operators.
/// ---------------------------------------------------------------------------
LLVM_ATTRIBUTE_ALWAYS_INLINE bool MPInt::operator==(const MPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall()))
    return getSmall() == O.getSmall();
  return detail::SlowMPInt(*this) == detail::SlowMPInt(O);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool MPInt::operator!=(const MPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall()))
    return getSmall() != O.getSmall();
  return detail::SlowMPInt(*this) != detail::SlowMPInt(O);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool MPInt::operator>(const MPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall()))
    return getSmall() > O.getSmall();
  return detail::SlowMPInt(*this) > detail::SlowMPInt(O);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool MPInt::operator<(const MPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall()))
    return getSmall() < O.getSmall();
  return detail::SlowMPInt(*this) < detail::SlowMPInt(O);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool MPInt::operator<=(const MPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall()))
    return getSmall() <= O.getSmall();
  return detail::SlowMPInt(*this) <= detail::SlowMPInt(O);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool MPInt::operator>=(const MPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall()))
    return getSmall() >= O.getSmall();
  return detail::SlowMPInt(*this) >= detail::SlowMPInt(O);
}

/// ---------------------------------------------------------------------------
/// Arithmetic operators.
/// ---------------------------------------------------------------------------

LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt MPInt::operator+(const MPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall())) {
    MPInt Result;
    bool Overflow =
        detail::addOverflow(getSmall(), O.getSmall(), Result.getSmall());
    if (LLVM_LIKELY(!Overflow))
      return Result;
    return MPInt(detail::SlowMPInt(*this) + detail::SlowMPInt(O));
  }
  return MPInt(detail::SlowMPInt(*this) + detail::SlowMPInt(O));
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt MPInt::operator-(const MPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall())) {
    MPInt Result;
    bool Overflow =
        detail::subOverflow(getSmall(), O.getSmall(), Result.getSmall());
    if (LLVM_LIKELY(!Overflow))
      return Result;
    return MPInt(detail::SlowMPInt(*this) - detail::SlowMPInt(O));
  }
  return MPInt(detail::SlowMPInt(*this) - detail::SlowMPInt(O));
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt MPInt::operator*(const MPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall())) {
    MPInt Result;
    bool Overflow =
        detail::mulOverflow(getSmall(), O.getSmall(), Result.getSmall());
    if (LLVM_LIKELY(!Overflow))
      return Result;
    return MPInt(detail::SlowMPInt(*this) * detail::SlowMPInt(O));
  }
  return MPInt(detail::SlowMPInt(*this) * detail::SlowMPInt(O));
}

// Division overflows only occur when negating the minimal possible value.
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt MPInt::divByPositive(const MPInt &O) const {
  assert(O > 0);
  if (LLVM_LIKELY(isSmall() && O.isSmall()))
    return MPInt(getSmall() / O.getSmall());
  return MPInt(detail::SlowMPInt(*this) / detail::SlowMPInt(O));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt MPInt::operator/(const MPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall())) {
    // Division overflows only occur when negating the minimal possible value.
    if (LLVM_UNLIKELY(detail::divWouldOverflow(getSmall(), O.getSmall())))
      return -*this;
    return MPInt(getSmall() / O.getSmall());
  }
  return MPInt(detail::SlowMPInt(*this) / detail::SlowMPInt(O));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt abs(const MPInt &X) {
  return MPInt(X >= 0 ? X : -X);
}
// Division overflows only occur when negating the minimal possible value.
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt ceilDiv(const MPInt &LHS, const MPInt &RHS) {
  if (LLVM_LIKELY(LHS.isSmall() && RHS.isSmall())) {
    if (LLVM_UNLIKELY(detail::divWouldOverflow(LHS.getSmall(), RHS.getSmall())))
      return -LHS;
    return MPInt(detail::ceilDiv(LHS.getSmall(), RHS.getSmall()));
  }
  return MPInt(ceilDiv(detail::SlowMPInt(LHS), detail::SlowMPInt(RHS)));
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt floorDiv(const MPInt &LHS,
                                            const MPInt &RHS) {
  if (LLVM_LIKELY(LHS.isSmall() && RHS.isSmall())) {
    if (LLVM_UNLIKELY(detail::divWouldOverflow(LHS.getSmall(), RHS.getSmall())))
      return -LHS;
    return MPInt(detail::floorDiv(LHS.getSmall(), RHS.getSmall()));
  }
  return MPInt(floorDiv(detail::SlowMPInt(LHS), detail::SlowMPInt(RHS)));
}
// The RHS is always expected to be positive, and the result
/// is always non-negative.
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt mod(const MPInt &LHS, const MPInt &RHS) {
  if (LLVM_LIKELY(LHS.isSmall() && RHS.isSmall()))
    return MPInt(detail::mod(LHS.getSmall(), RHS.getSmall()));
  return MPInt(mod(detail::SlowMPInt(LHS), detail::SlowMPInt(RHS)));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt gcd(const MPInt &A, const MPInt &B) {
  assert(A >= 0 && B >= 0 && "operands must be non-negative!");
  if (LLVM_LIKELY(A.isSmall() && B.isSmall()))
    return MPInt(std::gcd(A.getSmall(), B.getSmall()));
  return MPInt(gcd(detail::SlowMPInt(A), detail::SlowMPInt(B)));
}

/// Returns the least common multiple of A and B.
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt lcm(const MPInt &A, const MPInt &B) {
  MPInt X = abs(A);
  MPInt Y = abs(B);
  return (X * Y) / gcd(X, Y);
}

/// This operation cannot overflow.
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt MPInt::operator%(const MPInt &O) const {
  if (LLVM_LIKELY(isSmall() && O.isSmall()))
    return MPInt(getSmall() % O.getSmall());
  return MPInt(detail::SlowMPInt(*this) % detail::SlowMPInt(O));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt MPInt::operator-() const {
  if (LLVM_LIKELY(isSmall())) {
    if (LLVM_LIKELY(getSmall() != std::numeric_limits<int64_t>::min()))
      return MPInt(-getSmall());
    return MPInt(-detail::SlowMPInt(*this));
  }
  return MPInt(-detail::SlowMPInt(*this));
}

/// ---------------------------------------------------------------------------
/// Assignment operators, preincrement, predecrement.
/// ---------------------------------------------------------------------------
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt &MPInt::operator+=(const MPInt &O) {
  if (LLVM_LIKELY(isSmall() && O.isSmall())) {
    int64_t Result = getSmall();
    bool Overflow = detail::addOverflow(getSmall(), O.getSmall(), Result);
    if (LLVM_LIKELY(!Overflow)) {
      getSmall() = Result;
      return *this;
    }
    // Note: this return is not strictly required but
    // removing it leads to a performance regression.
    return *this = MPInt(detail::SlowMPInt(*this) + detail::SlowMPInt(O));
  }
  return *this = MPInt(detail::SlowMPInt(*this) + detail::SlowMPInt(O));
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt &MPInt::operator-=(const MPInt &O) {
  if (LLVM_LIKELY(isSmall() && O.isSmall())) {
    int64_t Result = getSmall();
    bool Overflow = detail::subOverflow(getSmall(), O.getSmall(), Result);
    if (LLVM_LIKELY(!Overflow)) {
      getSmall() = Result;
      return *this;
    }
    // Note: this return is not strictly required but
    // removing it leads to a performance regression.
    return *this = MPInt(detail::SlowMPInt(*this) - detail::SlowMPInt(O));
  }
  return *this = MPInt(detail::SlowMPInt(*this) - detail::SlowMPInt(O));
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt &MPInt::operator*=(const MPInt &O) {
  if (LLVM_LIKELY(isSmall() && O.isSmall())) {
    int64_t Result = getSmall();
    bool Overflow = detail::mulOverflow(getSmall(), O.getSmall(), Result);
    if (LLVM_LIKELY(!Overflow)) {
      getSmall() = Result;
      return *this;
    }
    // Note: this return is not strictly required but
    // removing it leads to a performance regression.
    return *this = MPInt(detail::SlowMPInt(*this) * detail::SlowMPInt(O));
  }
  return *this = MPInt(detail::SlowMPInt(*this) * detail::SlowMPInt(O));
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt &MPInt::operator/=(const MPInt &O) {
  if (LLVM_LIKELY(isSmall() && O.isSmall())) {
    // Division overflows only occur when negating the minimal possible value.
    if (LLVM_UNLIKELY(detail::divWouldOverflow(getSmall(), O.getSmall())))
      return *this = -*this;
    getSmall() /= O.getSmall();
    return *this;
  }
  return *this = MPInt(detail::SlowMPInt(*this) / detail::SlowMPInt(O));
}

// Division overflows only occur when the divisor is -1.
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt &
MPInt::divByPositiveInPlace(const MPInt &O) {
  assert(O > 0);
  if (LLVM_LIKELY(isSmall() && O.isSmall())) {
    getSmall() /= O.getSmall();
    return *this;
  }
  return *this = MPInt(detail::SlowMPInt(*this) / detail::SlowMPInt(O));
}

LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt &MPInt::operator%=(const MPInt &O) {
  return *this = *this % O;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt &MPInt::operator++() { return *this += 1; }
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt &MPInt::operator--() { return *this -= 1; }

/// ----------------------------------------------------------------------------
/// Convenience operator overloads for int64_t.
/// ----------------------------------------------------------------------------
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt &operator+=(MPInt &A, int64_t B) {
  return A = A + B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt &operator-=(MPInt &A, int64_t B) {
  return A = A - B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt &operator*=(MPInt &A, int64_t B) {
  return A = A * B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt &operator/=(MPInt &A, int64_t B) {
  return A = A / B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt &operator%=(MPInt &A, int64_t B) {
  return A = A % B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt operator+(const MPInt &A, int64_t B) {
  return A + MPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt operator-(const MPInt &A, int64_t B) {
  return A - MPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt operator*(const MPInt &A, int64_t B) {
  return A * MPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt operator/(const MPInt &A, int64_t B) {
  return A / MPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt operator%(const MPInt &A, int64_t B) {
  return A % MPInt(B);
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt operator+(int64_t A, const MPInt &B) {
  return MPInt(A) + B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt operator-(int64_t A, const MPInt &B) {
  return MPInt(A) - B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt operator*(int64_t A, const MPInt &B) {
  return MPInt(A) * B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt operator/(int64_t A, const MPInt &B) {
  return MPInt(A) / B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE MPInt operator%(int64_t A, const MPInt &B) {
  return MPInt(A) % B;
}

/// We provide special implementations of the comparison operators rather than
/// calling through as above, as this would result in a 1.2x slowdown.
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator==(const MPInt &A, int64_t B) {
  if (LLVM_LIKELY(A.isSmall()))
    return A.getSmall() == B;
  return A.getLarge() == B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator!=(const MPInt &A, int64_t B) {
  if (LLVM_LIKELY(A.isSmall()))
    return A.getSmall() != B;
  return A.getLarge() != B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator>(const MPInt &A, int64_t B) {
  if (LLVM_LIKELY(A.isSmall()))
    return A.getSmall() > B;
  return A.getLarge() > B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator<(const MPInt &A, int64_t B) {
  if (LLVM_LIKELY(A.isSmall()))
    return A.getSmall() < B;
  return A.getLarge() < B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator<=(const MPInt &A, int64_t B) {
  if (LLVM_LIKELY(A.isSmall()))
    return A.getSmall() <= B;
  return A.getLarge() <= B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator>=(const MPInt &A, int64_t B) {
  if (LLVM_LIKELY(A.isSmall()))
    return A.getSmall() >= B;
  return A.getLarge() >= B;
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator==(int64_t A, const MPInt &B) {
  if (LLVM_LIKELY(B.isSmall()))
    return A == B.getSmall();
  return A == B.getLarge();
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator!=(int64_t A, const MPInt &B) {
  if (LLVM_LIKELY(B.isSmall()))
    return A != B.getSmall();
  return A != B.getLarge();
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator>(int64_t A, const MPInt &B) {
  if (LLVM_LIKELY(B.isSmall()))
    return A > B.getSmall();
  return A > B.getLarge();
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator<(int64_t A, const MPInt &B) {
  if (LLVM_LIKELY(B.isSmall()))
    return A < B.getSmall();
  return A < B.getLarge();
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator<=(int64_t A, const MPInt &B) {
  if (LLVM_LIKELY(B.isSmall()))
    return A <= B.getSmall();
  return A <= B.getLarge();
}
LLVM_ATTRIBUTE_ALWAYS_INLINE bool operator>=(int64_t A, const MPInt &B) {
  if (LLVM_LIKELY(B.isSmall()))
    return A >= B.getSmall();
  return A >= B.getLarge();
}
} // namespace llvm

#endif // LLVM_ADT_MPINT_H
