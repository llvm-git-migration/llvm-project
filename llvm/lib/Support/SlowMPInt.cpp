//===- SlowMPInt.cpp - SlowMPInt Class ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SlowMPInt.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace detail;

SlowMPInt::SlowMPInt(int64_t Val) : Val(64, Val, /*isSigned=*/true) {}
SlowMPInt::SlowMPInt() : SlowMPInt(0) {}
SlowMPInt::SlowMPInt(const APInt &Val) : Val(Val) {}
SlowMPInt &SlowMPInt::operator=(int64_t Val) { return *this = SlowMPInt(Val); }
SlowMPInt::operator int64_t() const { return Val.getSExtValue(); }

hash_code detail::hash_value(const SlowMPInt &X) { return hash_value(X.Val); }

/// ---------------------------------------------------------------------------
/// Convenience operator overloads for int64_t.
/// ---------------------------------------------------------------------------
SlowMPInt &detail::operator+=(SlowMPInt &A, int64_t B) {
  return A += SlowMPInt(B);
}
SlowMPInt &detail::operator-=(SlowMPInt &A, int64_t B) {
  return A -= SlowMPInt(B);
}
SlowMPInt &detail::operator*=(SlowMPInt &A, int64_t B) {
  return A *= SlowMPInt(B);
}
SlowMPInt &detail::operator/=(SlowMPInt &A, int64_t B) {
  return A /= SlowMPInt(B);
}
SlowMPInt &detail::operator%=(SlowMPInt &A, int64_t B) {
  return A %= SlowMPInt(B);
}

bool detail::operator==(const SlowMPInt &A, int64_t B) {
  return A == SlowMPInt(B);
}
bool detail::operator!=(const SlowMPInt &A, int64_t B) {
  return A != SlowMPInt(B);
}
bool detail::operator>(const SlowMPInt &A, int64_t B) {
  return A > SlowMPInt(B);
}
bool detail::operator<(const SlowMPInt &A, int64_t B) {
  return A < SlowMPInt(B);
}
bool detail::operator<=(const SlowMPInt &A, int64_t B) {
  return A <= SlowMPInt(B);
}
bool detail::operator>=(const SlowMPInt &A, int64_t B) {
  return A >= SlowMPInt(B);
}
SlowMPInt detail::operator+(const SlowMPInt &A, int64_t B) {
  return A + SlowMPInt(B);
}
SlowMPInt detail::operator-(const SlowMPInt &A, int64_t B) {
  return A - SlowMPInt(B);
}
SlowMPInt detail::operator*(const SlowMPInt &A, int64_t B) {
  return A * SlowMPInt(B);
}
SlowMPInt detail::operator/(const SlowMPInt &A, int64_t B) {
  return A / SlowMPInt(B);
}
SlowMPInt detail::operator%(const SlowMPInt &A, int64_t B) {
  return A % SlowMPInt(B);
}

bool detail::operator==(int64_t A, const SlowMPInt &B) {
  return SlowMPInt(A) == B;
}
bool detail::operator!=(int64_t A, const SlowMPInt &B) {
  return SlowMPInt(A) != B;
}
bool detail::operator>(int64_t A, const SlowMPInt &B) {
  return SlowMPInt(A) > B;
}
bool detail::operator<(int64_t A, const SlowMPInt &B) {
  return SlowMPInt(A) < B;
}
bool detail::operator<=(int64_t A, const SlowMPInt &B) {
  return SlowMPInt(A) <= B;
}
bool detail::operator>=(int64_t A, const SlowMPInt &B) {
  return SlowMPInt(A) >= B;
}
SlowMPInt detail::operator+(int64_t A, const SlowMPInt &B) {
  return SlowMPInt(A) + B;
}
SlowMPInt detail::operator-(int64_t A, const SlowMPInt &B) {
  return SlowMPInt(A) - B;
}
SlowMPInt detail::operator*(int64_t A, const SlowMPInt &B) {
  return SlowMPInt(A) * B;
}
SlowMPInt detail::operator/(int64_t A, const SlowMPInt &B) {
  return SlowMPInt(A) / B;
}
SlowMPInt detail::operator%(int64_t A, const SlowMPInt &B) {
  return SlowMPInt(A) % B;
}

static unsigned getMaxWidth(const APInt &A, const APInt &B) {
  return std::max(A.getBitWidth(), B.getBitWidth());
}

/// ---------------------------------------------------------------------------
/// Comparison operators.
/// ---------------------------------------------------------------------------

// TODO: consider instead making APInt::compare available and using that.
bool SlowMPInt::operator==(const SlowMPInt &O) const {
  unsigned Width = getMaxWidth(Val, O.Val);
  return Val.sext(Width) == O.Val.sext(Width);
}
bool SlowMPInt::operator!=(const SlowMPInt &O) const {
  unsigned Width = getMaxWidth(Val, O.Val);
  return Val.sext(Width) != O.Val.sext(Width);
}
bool SlowMPInt::operator>(const SlowMPInt &O) const {
  unsigned Width = getMaxWidth(Val, O.Val);
  return Val.sext(Width).sgt(O.Val.sext(Width));
}
bool SlowMPInt::operator<(const SlowMPInt &O) const {
  unsigned Width = getMaxWidth(Val, O.Val);
  return Val.sext(Width).slt(O.Val.sext(Width));
}
bool SlowMPInt::operator<=(const SlowMPInt &O) const {
  unsigned Width = getMaxWidth(Val, O.Val);
  return Val.sext(Width).sle(O.Val.sext(Width));
}
bool SlowMPInt::operator>=(const SlowMPInt &O) const {
  unsigned Width = getMaxWidth(Val, O.Val);
  return Val.sext(Width).sge(O.Val.sext(Width));
}

/// ---------------------------------------------------------------------------
/// Arithmetic operators.
/// ---------------------------------------------------------------------------

/// Bring a and b to have the same width and then call op(a, b, overflow).
/// If the overflow bit becomes set, resize a and b to double the width and
/// call op(a, b, overflow), returning its result. The operation with double
/// widths should not also overflow.
APInt runOpWithExpandOnOverflow(
    const APInt &A, const APInt &B,
    function_ref<APInt(const APInt &, const APInt &, bool &Overflow)> Op) {
  bool Overflow;
  unsigned Width = getMaxWidth(A, B);
  APInt Ret = Op(A.sext(Width), B.sext(Width), Overflow);
  if (!Overflow)
    return Ret;

  Width *= 2;
  Ret = Op(A.sext(Width), B.sext(Width), Overflow);
  assert(!Overflow && "double width should be sufficient to avoid overflow!");
  return Ret;
}

SlowMPInt SlowMPInt::operator+(const SlowMPInt &O) const {
  return SlowMPInt(
      runOpWithExpandOnOverflow(Val, O.Val, std::mem_fn(&APInt::sadd_ov)));
}
SlowMPInt SlowMPInt::operator-(const SlowMPInt &O) const {
  return SlowMPInt(
      runOpWithExpandOnOverflow(Val, O.Val, std::mem_fn(&APInt::ssub_ov)));
}
SlowMPInt SlowMPInt::operator*(const SlowMPInt &O) const {
  return SlowMPInt(
      runOpWithExpandOnOverflow(Val, O.Val, std::mem_fn(&APInt::smul_ov)));
}
SlowMPInt SlowMPInt::operator/(const SlowMPInt &O) const {
  return SlowMPInt(
      runOpWithExpandOnOverflow(Val, O.Val, std::mem_fn(&APInt::sdiv_ov)));
}
SlowMPInt detail::abs(const SlowMPInt &X) { return X >= 0 ? X : -X; }
SlowMPInt detail::ceilDiv(const SlowMPInt &LHS, const SlowMPInt &RHS) {
  if (RHS == -1)
    return -LHS;
  unsigned Width = getMaxWidth(LHS.Val, RHS.Val);
  return SlowMPInt(APIntOps::RoundingSDiv(
      LHS.Val.sext(Width), RHS.Val.sext(Width), APInt::Rounding::UP));
}
SlowMPInt detail::floorDiv(const SlowMPInt &LHS, const SlowMPInt &RHS) {
  if (RHS == -1)
    return -LHS;
  unsigned Width = getMaxWidth(LHS.Val, RHS.Val);
  return SlowMPInt(APIntOps::RoundingSDiv(
      LHS.Val.sext(Width), RHS.Val.sext(Width), APInt::Rounding::DOWN));
}
// The RHS is always expected to be positive, and the result
/// is always non-negative.
SlowMPInt detail::mod(const SlowMPInt &LHS, const SlowMPInt &RHS) {
  assert(RHS >= 1 && "mod is only supported for positive divisors!");
  return LHS % RHS < 0 ? LHS % RHS + RHS : LHS % RHS;
}

SlowMPInt detail::gcd(const SlowMPInt &A, const SlowMPInt &B) {
  assert(A >= 0 && B >= 0 && "operands must be non-negative!");
  unsigned Width = getMaxWidth(A.Val, B.Val);
  return SlowMPInt(
      APIntOps::GreatestCommonDivisor(A.Val.sext(Width), B.Val.sext(Width)));
}

/// Returns the least common multiple of A and B.
SlowMPInt detail::lcm(const SlowMPInt &A, const SlowMPInt &B) {
  SlowMPInt X = abs(A);
  SlowMPInt Y = abs(B);
  return (X * Y) / gcd(X, Y);
}

/// This operation cannot overflow.
SlowMPInt SlowMPInt::operator%(const SlowMPInt &O) const {
  unsigned Width = std::max(Val.getBitWidth(), O.Val.getBitWidth());
  return SlowMPInt(Val.sext(Width).srem(O.Val.sext(Width)));
}

SlowMPInt SlowMPInt::operator-() const {
  if (Val.isMinSignedValue()) {
    /// Overflow only occurs when the value is the minimum possible value.
    APInt Ret = Val.sext(2 * Val.getBitWidth());
    return SlowMPInt(-Ret);
  }
  return SlowMPInt(-Val);
}

/// ---------------------------------------------------------------------------
/// Assignment operators, preincrement, predecrement.
/// ---------------------------------------------------------------------------
SlowMPInt &SlowMPInt::operator+=(const SlowMPInt &O) {
  *this = *this + O;
  return *this;
}
SlowMPInt &SlowMPInt::operator-=(const SlowMPInt &O) {
  *this = *this - O;
  return *this;
}
SlowMPInt &SlowMPInt::operator*=(const SlowMPInt &O) {
  *this = *this * O;
  return *this;
}
SlowMPInt &SlowMPInt::operator/=(const SlowMPInt &O) {
  *this = *this / O;
  return *this;
}
SlowMPInt &SlowMPInt::operator%=(const SlowMPInt &O) {
  *this = *this % O;
  return *this;
}
SlowMPInt &SlowMPInt::operator++() {
  *this += 1;
  return *this;
}

SlowMPInt &SlowMPInt::operator--() {
  *this -= 1;
  return *this;
}

/// ---------------------------------------------------------------------------
/// Printing.
/// ---------------------------------------------------------------------------
#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void SlowMPInt::print(raw_ostream &OS) const { OS << Val; }

void SlowMPInt::dump() const { print(dbgs()); }
#endif
