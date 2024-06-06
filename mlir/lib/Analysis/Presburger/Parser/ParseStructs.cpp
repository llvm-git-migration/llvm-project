//===- ParseStructs.cpp - Presburger Parse Structrures ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ParseStructs class that the parser for the
// Presburger library parses into.
//
//===----------------------------------------------------------------------===//

#include "ParseStructs.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir::presburger;
using llvm::cast;
using llvm::dbgs;
using llvm::isa;

bool AffineExprImpl::isPureAffine() const {
  switch (getKind()) {
  case AffineExprKind::SymbolId:
  case AffineExprKind::DimId:
  case AffineExprKind::Constant:
    return true;
  case AffineExprKind::Add: {
    const auto &op = cast<AffineBinOpExpr>(*this);
    return op.getLHS()->isPureAffine() && op.getRHS()->isPureAffine();
  }
  case AffineExprKind::Mul: {
    const auto &op = cast<AffineBinOpExpr>(*this);
    return op.getLHS()->isPureAffine() && op.getRHS()->isPureAffine() &&
           (isa<AffineConstantExpr>(op.getLHS()) ||
            isa<AffineConstantExpr>(op.getRHS()));
  }
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    const auto &op = cast<AffineBinOpExpr>(*this);
    return op.getLHS()->isPureAffine() && isa<AffineConstantExpr>(op.getRHS());
  }
  }
  llvm_unreachable("Unknown AffineExpr");
}

bool AffineExprImpl::isSymbolicOrConstant() const {
  switch (getKind()) {
  case AffineExprKind::Constant:
  case AffineExprKind::SymbolId:
    return true;
  case AffineExprKind::DimId:
    return false;
  case AffineExprKind::Add:
  case AffineExprKind::Mul:
  case AffineExprKind::FloorDiv:
  case AffineExprKind::CeilDiv:
  case AffineExprKind::Mod: {
    const auto &expr = cast<AffineBinOpExpr>(*this);
    return expr.getLHS()->isSymbolicOrConstant() &&
           expr.getRHS()->isSymbolicOrConstant();
  }
  }
  llvm_unreachable("Unknown AffineExpr");
}

// Simplify the mul to the extent required by usage and the flattener.
static AffineExpr simplifyMul(AffineExpr &&lhs, AffineExpr &&rhs) {
  if (isa<AffineConstantExpr>(*lhs) && isa<AffineConstantExpr>(*rhs)) {
    auto lhsConst = cast<AffineConstantExpr>(*lhs);
    auto rhsConst = cast<AffineConstantExpr>(*rhs);
    return std::make_unique<AffineConstantExpr>(lhsConst.getValue() *
                                                rhsConst.getValue());
  }

  if (!lhs->isSymbolicOrConstant() && !rhs->isSymbolicOrConstant())
    return nullptr;

  // Canonicalize the mul expression so that the constant/symbolic term is the
  // RHS. If both the lhs and rhs are symbolic, swap them if the lhs is a
  // constant. (Note that a constant is trivially symbolic).
  if (!rhs->isSymbolicOrConstant() || isa<AffineConstantExpr>(lhs)) {
    // At least one of them has to be symbolic.
    return std::move(rhs) * std::move(lhs);
  }

  // At this point, if there was a constant, it would be on the right.

  // Multiplication with a one is a noop, return the other input.
  if (isa<AffineConstantExpr>(*rhs)) {
    auto rhsConst = cast<AffineConstantExpr>(*rhs);
    if (rhsConst.getValue() == 1)
      return lhs;
    // Multiplication with zero.
    if (rhsConst.getValue() == 0)
      return std::make_unique<AffineConstantExpr>(rhsConst);
  }

  return nullptr;
}

namespace mlir::presburger {
AffineExpr operator*(AffineExpr &&s, AffineExpr &&o) {
  if (AffineExpr simpl = simplifyMul(std::move(s), std::move(o)))
    return simpl;
  return std::make_unique<AffineBinOpExpr>(std::move(s), std::move(o),
                                           AffineExprKind::Mul);
}
} // namespace mlir::presburger

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
enum class BindingStrength {
  Weak,   // + and -
  Strong, // All other binary operators.
};

static void printAffineExpr(const AffineExprImpl &expr,
                            BindingStrength enclosingTightness) {
  const char *binopSpelling = nullptr;
  switch (expr.getKind()) {
  case AffineExprKind::SymbolId: {
    unsigned pos = cast<AffineSymbolExpr>(expr).getPosition();
    dbgs() << 's' << pos;
    return;
  }
  case AffineExprKind::DimId: {
    unsigned pos = cast<AffineDimExpr>(expr).getPosition();
    dbgs() << 'd' << pos;
    return;
  }
  case AffineExprKind::Constant:
    dbgs() << cast<AffineConstantExpr>(expr).getValue();
    return;
  case AffineExprKind::Add:
    binopSpelling = " + ";
    break;
  case AffineExprKind::Mul:
    binopSpelling = " * ";
    break;
  case AffineExprKind::FloorDiv:
    binopSpelling = " floordiv ";
    break;
  case AffineExprKind::CeilDiv:
    binopSpelling = " ceildiv ";
    break;
  case AffineExprKind::Mod:
    binopSpelling = " mod ";
    break;
  }

  const auto &binOp = cast<AffineBinOpExpr>(expr);
  const AffineExprImpl &lhsExpr = *binOp.getLHS();
  const AffineExprImpl &rhsExpr = *binOp.getRHS();

  // Handle tightly binding binary operators.
  if (binOp.getKind() != AffineExprKind::Add) {
    if (enclosingTightness == BindingStrength::Strong)
      dbgs() << '(';

    // Pretty print multiplication with -1.
    if (isa<AffineConstantExpr>(rhsExpr)) {
      const auto &rhsConst = cast<AffineConstantExpr>(rhsExpr);
      if (binOp.getKind() == AffineExprKind::Mul && rhsConst.getValue() == -1) {
        dbgs() << "-";
        printAffineExpr(lhsExpr, BindingStrength::Strong);
        if (enclosingTightness == BindingStrength::Strong)
          dbgs() << ')';
        return;
      }
    }
    printAffineExpr(lhsExpr, BindingStrength::Strong);

    dbgs() << binopSpelling;
    printAffineExpr(rhsExpr, BindingStrength::Strong);

    if (enclosingTightness == BindingStrength::Strong)
      dbgs() << ')';
    return;
  }

  // Print out special "pretty" forms for add.
  if (enclosingTightness == BindingStrength::Strong)
    dbgs() << '(';

  // Pretty print addition to a product that has a negative operand as a
  // subtraction.
  if (isa<AffineBinOpExpr>(rhsExpr)) {
    const auto &rhs = cast<AffineBinOpExpr>(rhsExpr);
    if (rhs.getKind() == AffineExprKind::Mul) {
      const AffineExprImpl &rrhsExpr = *rhs.getRHS();
      if (isa<AffineConstantExpr>(rrhsExpr)) {
        const auto &rrhs = cast<AffineConstantExpr>(rrhsExpr);
        if (rrhs.getValue() == -1) {
          printAffineExpr(lhsExpr, BindingStrength::Weak);
          dbgs() << " - ";
          if (rhs.getLHS()->getKind() == AffineExprKind::Add) {
            printAffineExpr(*rhs.getLHS(), BindingStrength::Strong);
          } else {
            printAffineExpr(*rhs.getLHS(), BindingStrength::Weak);
          }

          if (enclosingTightness == BindingStrength::Strong)
            dbgs() << ')';
          return;
        }

        if (rrhs.getValue() < -1) {
          printAffineExpr(lhsExpr, BindingStrength::Weak);
          dbgs() << " - ";
          printAffineExpr(*rhs.getLHS(), BindingStrength::Strong);
          dbgs() << " * " << -rrhs.getValue();
          if (enclosingTightness == BindingStrength::Strong)
            dbgs() << ')';
          return;
        }
      }
    }
  }

  // Pretty print addition to a negative number as a subtraction.
  if (isa<AffineConstantExpr>(rhsExpr)) {
    const auto &rhsConst = cast<AffineConstantExpr>(rhsExpr);
    if (rhsConst.getValue() < 0) {
      printAffineExpr(lhsExpr, BindingStrength::Weak);
      dbgs() << " - " << -rhsConst.getValue();
      if (enclosingTightness == BindingStrength::Strong)
        dbgs() << ')';
      return;
    }
  }

  printAffineExpr(lhsExpr, BindingStrength::Weak);

  dbgs() << " + ";
  printAffineExpr(rhsExpr, BindingStrength::Weak);

  if (enclosingTightness == BindingStrength::Strong)
    dbgs() << ')';
}

LLVM_DUMP_METHOD void AffineExprImpl::dump() const {
  printAffineExpr(*this, BindingStrength::Weak);
  dbgs() << '\n';
}

LLVM_DUMP_METHOD void AffineMap::dump() const {
  dbgs() << "NumDims = " << numDims << '\n';
  dbgs() << "NumSymbols = " << numSymbols << '\n';
  dbgs() << "Expressions:\n";
  for (const AffineExpr &e : getExprs())
    e->dump();
}

LLVM_DUMP_METHOD void IntegerSet::dump() const {
  dbgs() << "NumDims = " << numDims << '\n';
  dbgs() << "NumSymbols = " << numSymbols << '\n';
  dbgs() << "Constraints:\n";
  for (const AffineExpr &c : getConstraints())
    c->dump();
  dbgs() << "EqFlags:\n";
  for (bool e : getEqFlags())
    dbgs() << e << '\n';
}
#endif
