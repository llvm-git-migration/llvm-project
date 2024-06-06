//===- ParseStructs.h - Presburger Parse Structrures ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PARSER_PARSESTRUCTS_H
#define MLIR_ANALYSIS_PRESBURGER_PARSER_PARSESTRUCTS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cassert>
#include <cstdint>
#include <memory>

namespace mlir {
namespace presburger {
namespace detail {
using llvm::ArrayRef;
using llvm::SmallVector;
using llvm::SmallVectorImpl;

enum class AffineExprKind {
  Add,
  /// RHS of mul is always a constant or a symbolic expression.
  Mul,
  /// RHS of mod is always a constant or a symbolic expression with a positive
  /// value.
  Mod,
  /// RHS of floordiv is always a constant or a symbolic expression.
  FloorDiv,
  /// RHS of ceildiv is always a constant or a symbolic expression.
  CeilDiv,
  /// This is a marker for the last affine binary op. The range of binary
  /// op's is expected to be this element and earlier.
  LAST_BINOP = CeilDiv,
  /// Constant integer.
  Constant,
  /// Dimensional identifier.
  DimId,
  /// Symbolic identifier.
  SymbolId,
};

struct AffineExprImpl {
  explicit AffineExprImpl(AffineExprKind kind) : kind(kind) {}

  // Delete all copy/move operators.
  AffineExprImpl(const AffineExprImpl &o) = delete;
  AffineExprImpl &operator=(const AffineExprImpl &o) = delete;
  AffineExprImpl(AffineExprImpl &&o) = delete;
  AffineExprImpl &operator=(AffineExprImpl &&o) = delete;

  AffineExprKind getKind() const { return kind; }

  /// Returns true if this expression is made out of only symbols and
  /// constants, i.e., it does not involve dimensional identifiers.
  bool isSymbolicOrConstant() const;

  /// Returns true if this is a pure affine expression, i.e., multiplication,
  /// floordiv, ceildiv, and mod is only allowed w.r.t constants.
  bool isPureAffine() const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif

  AffineExprKind kind;
};

// AffineExpr is a unique ptr, since there is a cycle is AffineBinaryOp.
using AffineExpr = std::unique_ptr<AffineExprImpl>;

struct AffineBinOpExpr : public AffineExprImpl {
  AffineBinOpExpr(AffineExpr &&lhs, AffineExpr &&rhs, AffineExprKind kind)
      : AffineExprImpl(kind), lhs(std::move(lhs)), rhs(std::move(rhs)) {}

  // Delete all copy/move operators.
  AffineBinOpExpr(const AffineBinOpExpr &o) = delete;
  AffineBinOpExpr &operator=(const AffineBinOpExpr &o) = delete;
  AffineBinOpExpr(AffineBinOpExpr &&o) = delete;
  AffineBinOpExpr &operator=(AffineBinOpExpr &&o) = delete;

  const AffineExpr &getLHS() const { return lhs; }
  const AffineExpr &getRHS() const { return rhs; }
  static bool classof(const AffineExprImpl *a) {
    return a->getKind() <= AffineExprKind::LAST_BINOP;
  }

  AffineExpr lhs;
  AffineExpr rhs;
};

/// A dimensional or symbolic identifier appearing in an affine expression.
struct AffineDimExpr : public AffineExprImpl {
  AffineDimExpr(unsigned position)
      : AffineExprImpl(AffineExprKind::DimId), position(position) {}

  // Enable copy/move constructors; trivial.
  AffineDimExpr(const AffineDimExpr &o)
      : AffineExprImpl(AffineExprKind::DimId), position(o.position) {}
  AffineDimExpr(AffineDimExpr &&o)
      : AffineExprImpl(AffineExprKind::DimId), position(o.position) {}
  AffineDimExpr &operator=(const AffineDimExpr &o) = delete;
  AffineDimExpr &operator=(AffineDimExpr &&o) = delete;

  unsigned getPosition() const { return position; }
  static bool classof(const AffineExprImpl *a) {
    return a->getKind() == AffineExprKind::DimId;
  }
  bool operator==(const AffineDimExpr &o) const {
    return position == o.position;
  }

  /// Position of this identifier in the argument list.
  unsigned position;
};

/// A symbolic identifier appearing in an affine expression.
struct AffineSymbolExpr : public AffineExprImpl {
  AffineSymbolExpr(unsigned position)
      : AffineExprImpl(AffineExprKind::SymbolId), position(position) {}

  // Enable copy/move constructors; trivial.
  AffineSymbolExpr(const AffineSymbolExpr &o)
      : AffineExprImpl(AffineExprKind::SymbolId), position(o.position) {}
  AffineSymbolExpr(AffineSymbolExpr &&o)
      : AffineExprImpl(AffineExprKind::SymbolId), position(o.position) {}
  AffineSymbolExpr &operator=(const AffineSymbolExpr &o) = delete;
  AffineSymbolExpr &operator=(AffineSymbolExpr &&o) = delete;

  unsigned getPosition() const { return position; }
  static bool classof(const AffineExprImpl *a) {
    return a->getKind() == AffineExprKind::SymbolId;
  }
  bool operator==(const AffineSymbolExpr &o) const {
    return position == o.position;
  }

  /// Position of this identifier in the argument list.
  unsigned position;
};

/// An integer constant appearing in affine expression.
struct AffineConstantExpr : public AffineExprImpl {
  AffineConstantExpr(int64_t constant)
      : AffineExprImpl(AffineExprKind::Constant), constant(constant) {}

  // Enable copy/move constructors; trivial.
  AffineConstantExpr(const AffineConstantExpr &o)
      : AffineExprImpl(AffineExprKind::Constant), constant(o.constant) {}
  AffineConstantExpr(AffineConstantExpr &&o)
      : AffineExprImpl(AffineExprKind::Constant), constant(o.constant) {}
  AffineConstantExpr &operator=(const AffineConstantExpr &o) = delete;
  AffineConstantExpr &operator=(AffineConstantExpr &&o) = delete;

  int64_t getValue() const { return constant; }
  static bool classof(const AffineExprImpl *a) {
    return a->getKind() == AffineExprKind::Constant;
  }
  bool operator==(const AffineConstantExpr &o) const {
    return constant == o.constant;
  }

  // The constant.
  int64_t constant;
};

struct AffineMap {
  unsigned numDims;
  unsigned numSymbols;

  // The affine expressions in the map.
  SmallVector<AffineExpr, 4> exprs;

  AffineMap(unsigned numDims, unsigned numSymbols,
            SmallVectorImpl<AffineExpr> &&exprs)
      : numDims(numDims), numSymbols(numSymbols), exprs(std::move(exprs)) {}

  // Non-copyable; only movable.
  AffineMap(const AffineMap &) = delete;
  AffineMap operator=(const AffineMap &) = delete;
  AffineMap(AffineMap &&o)
      : numDims(o.numDims), numSymbols(o.numSymbols),
        exprs(std::move(o.exprs)) {}
  AffineMap &operator=(AffineMap &&o) = delete;

  unsigned getNumDims() const { return numDims; }
  unsigned getNumSymbols() const { return numSymbols; }
  unsigned getNumInputs() const { return numDims + numSymbols; }
  unsigned getNumExprs() const { return exprs.size(); }
  ArrayRef<AffineExpr> getExprs() const { return exprs; }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif
};

struct IntegerSet {
  unsigned numDims;
  unsigned numSymbols;

  /// Array of affine constraints: a constraint is either an equality
  /// (affine_expr == 0) or an inequality (affine_expr >= 0).
  SmallVector<AffineExpr, 4> constraints;

  // Bits to check whether a constraint is an equality or an inequality.
  SmallVector<bool, 4> eqFlags;

  IntegerSet(unsigned numDims, unsigned numSymbols,
             SmallVectorImpl<AffineExpr> &&constraints,
             SmallVectorImpl<bool> &&eqFlags)
      : numDims(numDims), numSymbols(numSymbols),
        constraints(std::move(constraints)), eqFlags(std::move(eqFlags)) {
    assert(constraints.size() == eqFlags.size());
  }

  // Non-copyable; only movable.
  IntegerSet(const IntegerSet &o) = delete;
  IntegerSet &operator=(const IntegerSet &o) = delete;
  IntegerSet(IntegerSet &&o)
      : numDims(o.numDims), numSymbols(o.numSymbols),
        constraints(std::move(o.constraints)), eqFlags(std::move(o.eqFlags)) {}
  IntegerSet &operator=(IntegerSet &&o) = delete;

  IntegerSet(unsigned dimCount, unsigned symbolCount, AffineExpr &&constraint,
             bool eqFlag)
      : numDims(dimCount), numSymbols(symbolCount) {
    constraints.emplace_back(std::move(constraint));
    eqFlags.emplace_back(eqFlag);
  }

  unsigned getNumDims() const { return numDims; }
  unsigned getNumSymbols() const { return numSymbols; }
  unsigned getNumInputs() const { return numDims + numSymbols; }
  ArrayRef<AffineExpr> getConstraints() const { return constraints; }
  unsigned getNumConstraints() const { return constraints.size(); }
  ArrayRef<bool> getEqFlags() const { return eqFlags; }
  bool isEq(unsigned idx) const { return eqFlags[idx]; };

  unsigned getNumEqualities() const {
    unsigned numEqualities = 0;
    for (unsigned i = 0, e = getNumConstraints(); i < e; i++)
      if (isEq(i))
        ++numEqualities;
    return numEqualities;
  }

  unsigned getNumInequalities() const {
    return getNumConstraints() - getNumEqualities();
  }

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif
};

// Convenience operators.
AffineExpr operator*(AffineExpr &&s, AffineExpr &&o);
inline AffineExpr operator*(AffineExpr &&s, int64_t o) {
  return std::move(s) * std::make_unique<AffineConstantExpr>(o);
}
inline AffineExpr operator*(int64_t s, AffineExpr &&o) {
  return std::move(o) * s;
}
inline AffineExpr operator+(AffineExpr &&s, AffineExpr &&o) {
  return std::make_unique<AffineBinOpExpr>(std::move(s), std::move(o),
                                           AffineExprKind::Add);
}
inline AffineExpr operator+(AffineExpr &&s, int64_t o) {
  return std::move(s) + std::make_unique<AffineConstantExpr>(o);
}
inline AffineExpr operator+(int64_t s, AffineExpr &&o) {
  return std::move(o) + s;
}
inline AffineExpr operator-(AffineExpr &&s, AffineExpr &&o) {
  return std::move(s) + std::move(o) * -1;
}
inline AffineExpr operator%(AffineExpr &&s, AffineExpr &&o) {
  return std::make_unique<AffineBinOpExpr>(std::move(s), std::move(o),
                                           AffineExprKind::Mod);
}
inline AffineExpr ceilDiv(AffineExpr &&s, AffineExpr &&o) {
  return std::make_unique<AffineBinOpExpr>(std::move(s), std::move(o),
                                           AffineExprKind::CeilDiv);
}
inline AffineExpr floorDiv(AffineExpr &&s, AffineExpr &&o) {
  return std::make_unique<AffineBinOpExpr>(std::move(s), std::move(o),
                                           AffineExprKind::FloorDiv);
}
} // namespace detail

using AffineMap = detail::AffineMap;
using IntegerSet = detail::IntegerSet;
} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PARSER_PARSESTRUCTS_H
