//===- SlowMPInt.h - SlowMPInt Class ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a simple class to represent arbitrary precision signed integers.
// Unlike APInt, one does not have to specify a fixed maximum size, and the
// integer can take on any arbitrary values.
//
// This class is to be used as a fallback slow path for the MPInt class, and
// is not intended to be used directly.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SLOWMPINT_H
#define LLVM_ADT_SLOWMPINT_H

#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace detail {
/// A simple class providing multi-precision arithmetic. Internally, it stores
/// an APInt, whose width is doubled whenever an overflow occurs at a certain
/// width. The default constructor sets the initial width to 64. SlowMPInt is
/// primarily intended to be used as a slow fallback path for the upcoming MPInt
/// class.
class SlowMPInt {
private:
  APInt Val;

public:
  explicit SlowMPInt(int64_t Val);
  SlowMPInt();
  explicit SlowMPInt(const APInt &Val);
  SlowMPInt &operator=(int64_t Val);
  explicit operator int64_t() const;
  SlowMPInt operator-() const;
  bool operator==(const SlowMPInt &O) const;
  bool operator!=(const SlowMPInt &O) const;
  bool operator>(const SlowMPInt &O) const;
  bool operator<(const SlowMPInt &O) const;
  bool operator<=(const SlowMPInt &O) const;
  bool operator>=(const SlowMPInt &O) const;
  SlowMPInt operator+(const SlowMPInt &O) const;
  SlowMPInt operator-(const SlowMPInt &O) const;
  SlowMPInt operator*(const SlowMPInt &O) const;
  SlowMPInt operator/(const SlowMPInt &O) const;
  SlowMPInt operator%(const SlowMPInt &O) const;
  SlowMPInt &operator+=(const SlowMPInt &O);
  SlowMPInt &operator-=(const SlowMPInt &O);
  SlowMPInt &operator*=(const SlowMPInt &O);
  SlowMPInt &operator/=(const SlowMPInt &O);
  SlowMPInt &operator%=(const SlowMPInt &O);

  SlowMPInt &operator++();
  SlowMPInt &operator--();

  friend SlowMPInt abs(const SlowMPInt &X);
  friend SlowMPInt ceilDiv(const SlowMPInt &LHS, const SlowMPInt &RHS);
  friend SlowMPInt floorDiv(const SlowMPInt &LHS, const SlowMPInt &RHS);
  /// The operands must be non-negative for gcd.
  friend SlowMPInt gcd(const SlowMPInt &A, const SlowMPInt &B);

  /// Overload to compute a hash_code for a SlowMPInt value.
  friend hash_code hash_value(const SlowMPInt &X); // NOLINT

  unsigned getBitWidth() const { return Val.getBitWidth(); }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  void print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif
};

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
inline raw_ostream &operator<<(raw_ostream &OS, const SlowMPInt &X) {
  X.print(OS);
  return OS;
}
#endif

/// Returns the remainder of dividing LHS by RHS.
///
/// The RHS is always expected to be positive, and the result
/// is always non-negative.
SlowMPInt mod(const SlowMPInt &LHS, const SlowMPInt &RHS);

/// Returns the least common multiple of A and B.
SlowMPInt lcm(const SlowMPInt &A, const SlowMPInt &B);

/// Redeclarations of friend declarations above to
/// make it discoverable by lookups.
SlowMPInt abs(const SlowMPInt &X);
SlowMPInt ceilDiv(const SlowMPInt &LHS, const SlowMPInt &RHS);
SlowMPInt floorDiv(const SlowMPInt &LHS, const SlowMPInt &RHS);
SlowMPInt gcd(const SlowMPInt &A, const SlowMPInt &B);
hash_code hash_value(const SlowMPInt &X); // NOLINT

/// ---------------------------------------------------------------------------
/// Convenience operator overloads for int64_t.
/// ---------------------------------------------------------------------------
SlowMPInt &operator+=(SlowMPInt &A, int64_t B);
SlowMPInt &operator-=(SlowMPInt &A, int64_t B);
SlowMPInt &operator*=(SlowMPInt &A, int64_t B);
SlowMPInt &operator/=(SlowMPInt &A, int64_t B);
SlowMPInt &operator%=(SlowMPInt &A, int64_t B);

bool operator==(const SlowMPInt &A, int64_t B);
bool operator!=(const SlowMPInt &A, int64_t B);
bool operator>(const SlowMPInt &A, int64_t B);
bool operator<(const SlowMPInt &A, int64_t B);
bool operator<=(const SlowMPInt &A, int64_t B);
bool operator>=(const SlowMPInt &A, int64_t B);
SlowMPInt operator+(const SlowMPInt &A, int64_t B);
SlowMPInt operator-(const SlowMPInt &A, int64_t B);
SlowMPInt operator*(const SlowMPInt &A, int64_t B);
SlowMPInt operator/(const SlowMPInt &A, int64_t B);
SlowMPInt operator%(const SlowMPInt &A, int64_t B);

bool operator==(int64_t A, const SlowMPInt &B);
bool operator!=(int64_t A, const SlowMPInt &B);
bool operator>(int64_t A, const SlowMPInt &B);
bool operator<(int64_t A, const SlowMPInt &B);
bool operator<=(int64_t A, const SlowMPInt &B);
bool operator>=(int64_t A, const SlowMPInt &B);
SlowMPInt operator+(int64_t A, const SlowMPInt &B);
SlowMPInt operator-(int64_t A, const SlowMPInt &B);
SlowMPInt operator*(int64_t A, const SlowMPInt &B);
SlowMPInt operator/(int64_t A, const SlowMPInt &B);
SlowMPInt operator%(int64_t A, const SlowMPInt &B);
} // namespace detail
} // namespace llvm

#endif // LLVM_ADT_SLOWMPINT_H
