//===- Flattener.cpp - Presburger ParseStruct Flattener ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Flattener class for flattening the parse tree
// produced by the parser for the Presburger library.
//
//===----------------------------------------------------------------------===//

#include "Flattener.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace presburger {
namespace detail {
using llvm::SmallVector;

AffineExpr AffineExprFlattener::getAffineExprFromFlatForm(
    ArrayRef<int64_t> flatExprs, unsigned numDims, unsigned numSymbols) {
  assert(flatExprs.size() - numDims - numSymbols - 1 == localExprs.size() &&
         "unexpected number of local expressions");

  // Dimensions and symbols.
  AffineExpr expr = std::make_unique<AffineConstantExpr>(0);
  for (unsigned j = 0; j < getLocalVarStartIndex(); ++j) {
    if (flatExprs[j] == 0)
      continue;
    if (j < numDims)
      expr =
          std::move(expr) + std::make_unique<AffineDimExpr>(j) * flatExprs[j];
    else
      expr = std::move(expr) +
             std::make_unique<AffineSymbolExpr>(j - numDims) * flatExprs[j];
  }

  // Local identifiers.
  for (unsigned j = getLocalVarStartIndex(); j < flatExprs.size() - 1; ++j) {
    if (flatExprs[j] == 0)
      continue;
    // It is safe to move out of the localExprs vector, since no expr is used
    // more than once.
    AffineExpr term =
        std::move(localExprs[j - getLocalVarStartIndex()]) * flatExprs[j];
    expr = std::move(expr) + std::move(term);
  }

  // Constant term.
  int64_t constTerm = flatExprs[flatExprs.size() - 1];
  if (constTerm != 0)
    return std::move(expr) + constTerm;
  return expr;
}

// In pure affine t = expr * c, we multiply each coefficient of lhs with c.
//
// In case of semi affine multiplication expressions, t = expr * symbolic_expr,
// introduce a local variable p (= expr * symbolic_expr), and the affine
// expression expr * symbolic_expr is added to `localExprs`.
LogicalResult AffineExprFlattener::visitMulExpr(const AffineBinOpExpr &expr) {
  assert(operandExprStack.size() >= 2);
  SmallVector<int64_t, 8> rhs = operandExprStack.back();
  operandExprStack.pop_back();
  SmallVector<int64_t, 8> &lhs = operandExprStack.back();

  // Flatten semi-affine multiplication expressions by introducing a local
  // variable in place of the product; the affine expression
  // corresponding to the quantifier is added to `localExprs`.
  if (!isa<AffineConstantExpr>(expr.getRHS())) {
    AffineExpr a = getAffineExprFromFlatForm(lhs, numDims, numSymbols);
    AffineExpr b = getAffineExprFromFlatForm(rhs, numDims, numSymbols);
    addLocalVariableSemiAffine(std::move(a) * std::move(b), lhs, lhs.size());
    return success();
  }

  // Get the RHS constant.
  int64_t rhsConst = rhs[getConstantIndex()];
  for (int64_t &lhsElt : lhs)
    lhsElt *= rhsConst;

  return success();
}

LogicalResult AffineExprFlattener::visitAddExpr(const AffineBinOpExpr &expr) {
  assert(operandExprStack.size() >= 2);
  const auto &rhs = operandExprStack.back();
  auto &lhs = operandExprStack[operandExprStack.size() - 2];
  assert(lhs.size() == rhs.size());
  // Update the LHS in place.
  for (unsigned i = 0; i < rhs.size(); ++i)
    lhs[i] += rhs[i];
  // Pop off the RHS.
  operandExprStack.pop_back();
  return success();
}

//
// t = expr mod c   <=>  t = expr - c*q and c*q <= expr <= c*q + c - 1
//
// A mod expression "expr mod c" is thus flattened by introducing a new local
// variable q (= expr floordiv c), such that expr mod c is replaced with
// 'expr - c * q' and c * q <= expr <= c * q + c - 1 are added to localVarCst.
//
// In case of semi-affine modulo expressions, t = expr mod symbolic_expr,
// introduce a local variable m (= expr mod symbolic_expr), and the affine
// expression expr mod symbolic_expr is added to `localExprs`.
LogicalResult AffineExprFlattener::visitModExpr(const AffineBinOpExpr &expr) {
  assert(operandExprStack.size() >= 2);

  SmallVector<int64_t, 8> rhs = operandExprStack.back();
  operandExprStack.pop_back();
  SmallVector<int64_t, 8> &lhs = operandExprStack.back();

  // Flatten semi affine modulo expressions by introducing a local
  // variable in place of the modulo value, and the affine expression
  // corresponding to the quantifier is added to `localExprs`.
  if (!isa<AffineConstantExpr>(expr.getRHS())) {
    AffineExpr dividendExpr =
        getAffineExprFromFlatForm(lhs, numDims, numSymbols);
    AffineExpr divisorExpr =
        getAffineExprFromFlatForm(rhs, numDims, numSymbols);
    AffineExpr modExpr = std::move(dividendExpr) % std::move(divisorExpr);
    addLocalVariableSemiAffine(std::move(modExpr), lhs, lhs.size());
    return success();
  }

  int64_t rhsConst = rhs[getConstantIndex()];
  if (rhsConst <= 0)
    return failure();

  // Check if the LHS expression is a multiple of modulo factor.
  unsigned i;
  for (i = 0; i < lhs.size(); ++i)
    if (lhs[i] % rhsConst != 0)
      break;
  // If yes, modulo expression here simplifies to zero.
  if (i == lhs.size()) {
    std::fill(lhs.begin(), lhs.end(), 0);
    return success();
  }

  // Add a local variable for the quotient, i.e., expr % c is replaced by
  // (expr - q * c) where q = expr floordiv c. Do this while canceling out
  // the GCD of expr and c.
  SmallVector<int64_t, 8> floorDividend(lhs);
  uint64_t gcd = rhsConst;
  for (int64_t lhsElt : lhs)
    gcd = std::gcd(gcd, (uint64_t)std::abs(lhsElt));
  // Simplify the numerator and the denominator.
  if (gcd != 1) {
    for (int64_t &floorDividendElt : floorDividend)
      floorDividendElt = floorDividendElt / static_cast<int64_t>(gcd);
  }
  int64_t floorDivisor = rhsConst / static_cast<int64_t>(gcd);

  // Construct the AffineExpr form of the floordiv to store in localExprs.

  AffineExpr dividendExpr =
      getAffineExprFromFlatForm(floorDividend, numDims, numSymbols);
  AffineExpr divisorExpr = std::make_unique<AffineConstantExpr>(floorDivisor);
  AffineExpr floorDivExpr =
      floorDiv(std::move(dividendExpr), std::move(divisorExpr));
  int loc;
  if ((loc = findLocalId(floorDivExpr)) == -1) {
    addLocalFloorDivId(floorDividend, floorDivisor, std::move(floorDivExpr));
    // Set result at top of stack to "lhs - rhsConst * q".
    lhs[getLocalVarStartIndex() + numLocals - 1] = -rhsConst;
  } else {
    // Reuse the existing local id.
    lhs[getLocalVarStartIndex() + loc] = -rhsConst;
  }
  return success();
}

LogicalResult
AffineExprFlattener::visitCeilDivExpr(const AffineBinOpExpr &expr) {
  return visitDivExpr(expr, /*isCeil=*/true);
}
LogicalResult
AffineExprFlattener::visitFloorDivExpr(const AffineBinOpExpr &expr) {
  return visitDivExpr(expr, /*isCeil=*/false);
}

LogicalResult AffineExprFlattener::visitDimExpr(const AffineDimExpr &expr) {
  operandExprStack.emplace_back(SmallVector<int64_t, 32>(getNumCols(), 0));
  auto &eq = operandExprStack.back();
  assert(expr.getPosition() < numDims && "Inconsistent number of dims");
  eq[getDimStartIndex() + expr.getPosition()] = 1;
  return success();
}

LogicalResult
AffineExprFlattener::visitSymbolExpr(const AffineSymbolExpr &expr) {
  operandExprStack.emplace_back(SmallVector<int64_t, 32>(getNumCols(), 0));
  auto &eq = operandExprStack.back();
  assert(expr.getPosition() < numSymbols && "inconsistent number of symbols");
  eq[getSymbolStartIndex() + expr.getPosition()] = 1;
  return success();
}

LogicalResult
AffineExprFlattener::visitConstantExpr(const AffineConstantExpr &expr) {
  operandExprStack.emplace_back(SmallVector<int64_t, 32>(getNumCols(), 0));
  auto &eq = operandExprStack.back();
  eq[getConstantIndex()] = expr.getValue();
  return success();
}

void AffineExprFlattener::addLocalVariableSemiAffine(
    AffineExpr &&expr, SmallVectorImpl<int64_t> &result,
    unsigned long resultSize) {
  assert(result.size() == resultSize && "result vector size mismatch");
  int loc;
  if ((loc = findLocalId(expr)) == -1)
    addLocalIdSemiAffine(std::move(expr));
  std::fill(result.begin(), result.end(), 0);
  if (loc == -1)
    result[getLocalVarStartIndex() + numLocals - 1] = 1;
  else
    result[getLocalVarStartIndex() + loc] = 1;
}

// t = expr floordiv c   <=> t = q, c * q <= expr <= c * q + c - 1
// A floordiv is thus flattened by introducing a new local variable q, and
// replacing that expression with 'q' while adding the constraints
// c * q <= expr <= c * q + c - 1 to localVarCst (done by
// IntegerRelation::addLocalFloorDiv).
//
// A ceildiv is similarly flattened:
// t = expr ceildiv c   <=> t =  (expr + c - 1) floordiv c
//
// In case of semi affine division expressions, t = expr floordiv symbolic_expr
// or t = expr ceildiv symbolic_expr, introduce a local variable q (= expr
// floordiv/ceildiv symbolic_expr), and the affine floordiv/ceildiv is added to
// `localExprs`.
LogicalResult AffineExprFlattener::visitDivExpr(const AffineBinOpExpr &expr,
                                                bool isCeil) {
  assert(operandExprStack.size() >= 2);

  SmallVector<int64_t, 8> rhs = operandExprStack.back();
  operandExprStack.pop_back();
  SmallVector<int64_t, 8> &lhs = operandExprStack.back();

  // Flatten semi affine division expressions by introducing a local
  // variable in place of the quotient, and the affine expression corresponding
  // to the quantifier is added to `localExprs`.
  if (!isa<AffineConstantExpr>(expr.getRHS())) {
    AffineExpr a = getAffineExprFromFlatForm(lhs, numDims, numSymbols);
    AffineExpr b = getAffineExprFromFlatForm(rhs, numDims, numSymbols);
    AffineExpr divExpr = isCeil ? ceilDiv(std::move(a), std::move(b))
                                : floorDiv(std::move(a), std::move(b));
    addLocalVariableSemiAffine(std::move(divExpr), lhs, lhs.size());
    return success();
  }

  // This is a pure affine expr; the RHS is a positive constant.
  int64_t rhsConst = rhs[getConstantIndex()];
  if (rhsConst <= 0)
    return failure();

  // Simplify the floordiv, ceildiv if possible by canceling out the greatest
  // common divisors of the numerator and denominator.
  uint64_t gcd = std::abs(rhsConst);
  for (int64_t lhsElt : lhs)
    gcd = std::gcd(gcd, (uint64_t)std::abs(lhsElt));
  // Simplify the numerator and the denominator.
  if (gcd != 1) {
    for (int64_t &lhsElt : lhs)
      lhsElt = lhsElt / static_cast<int64_t>(gcd);
  }
  int64_t divisor = rhsConst / static_cast<int64_t>(gcd);
  // If the divisor becomes 1, the updated LHS is the result. (The
  // divisor can't be negative since rhsConst is positive).
  if (divisor == 1)
    return success();

  // If the divisor cannot be simplified to one, we will have to retain
  // the ceil/floor expr (simplified up until here). Add an existential
  // quantifier to express its result, i.e., expr1 div expr2 is replaced
  // by a new identifier, q.
  AffineExpr a = getAffineExprFromFlatForm(lhs, numDims, numSymbols);
  AffineExpr b = std::make_unique<AffineConstantExpr>(divisor);

  int loc;
  AffineExpr divExpr = isCeil ? ceilDiv(std::move(a), std::move(b))
                              : floorDiv(std::move(a), std::move(b));
  if ((loc = findLocalId(divExpr)) == -1) {
    if (!isCeil) {
      SmallVector<int64_t, 8> dividend(lhs);
      addLocalFloorDivId(dividend, divisor, std::move(divExpr));
    } else {
      // lhs ceildiv c <=>  (lhs + c - 1) floordiv c
      SmallVector<int64_t, 8> dividend(lhs);
      dividend.back() += divisor - 1;
      addLocalFloorDivId(dividend, divisor, std::move(divExpr));
    }
  }
  // Set the expression on stack to the local var introduced to capture the
  // result of the division (floor or ceil).
  std::fill(lhs.begin(), lhs.end(), 0);
  if (loc == -1)
    lhs[getLocalVarStartIndex() + numLocals - 1] = 1;
  else
    lhs[getLocalVarStartIndex() + loc] = 1;
  return success();
}

void AffineExprFlattener::addLocalFloorDivId(ArrayRef<int64_t> dividend,
                                             int64_t divisor,
                                             AffineExpr &&localExpr) {
  assert(divisor > 0 && "positive constant divisor expected");
  for (SmallVector<int64_t, 8> &subExpr : operandExprStack)
    subExpr.insert(subExpr.begin() + getLocalVarStartIndex() + numLocals, 0);
  localExprs.emplace_back(std::move(localExpr));
  ++numLocals;
  // Update localVarCst.
  localVarCst.addLocalFloorDiv(dividend, divisor);
}

void AffineExprFlattener::addLocalIdSemiAffine(AffineExpr &&localExpr) {
  for (SmallVector<int64_t, 8> &subExpr : operandExprStack)
    subExpr.insert(subExpr.begin() + getLocalVarStartIndex() + numLocals, 0);
  localExprs.emplace_back(std::move(localExpr));
  ++numLocals;
}

int AffineExprFlattener::findLocalId(const AffineExpr &localExpr) {
  auto *it = llvm::find(localExprs, localExpr);
  if (it == localExprs.end())
    return -1;
  return it - localExprs.begin();
}

AffineExprFlattener::AffineExprFlattener(unsigned numDims, unsigned numSymbols)
    : numDims(numDims), numSymbols(numSymbols), numLocals(0),
      localVarCst(PresburgerSpace::getSetSpace(numDims, numSymbols)) {
  operandExprStack.reserve(8);
}

// Flattens the expressions in map. Returns failure if 'expr' was unable to be
// flattened. For example two specific cases:
// 1. semi-affine expressions not handled yet.
// 2. has poison expression (i.e., division by zero).
static LogicalResult
getFlattenedAffineExprs(ArrayRef<AffineExpr> exprs, unsigned numDims,
                        unsigned numSymbols,
                        std::vector<SmallVector<int64_t, 8>> &flattenedExprs,
                        IntegerPolyhedron &localVarCst) {
  if (exprs.empty()) {
    localVarCst = IntegerPolyhedron(
        0, 0, numDims + numSymbols + 1,
        presburger::PresburgerSpace::getSetSpace(numDims, numSymbols, 0));
    return success();
  }

  AffineExprFlattener flattener(numDims, numSymbols);
  // Use the same flattener to simplify each expression successively. This way
  // local variables / expressions are shared.
  for (const AffineExpr &expr : exprs) {
    if (!expr->isPureAffine())
      return failure();
    // has poison expression
    LogicalResult flattenResult = flattener.walkPostOrder(*expr);
    if (failed(flattenResult))
      return failure();
  }

  assert(flattener.operandExprStack.size() == exprs.size());
  flattenedExprs.clear();
  flattenedExprs.assign(flattener.operandExprStack.begin(),
                        flattener.operandExprStack.end());

  localVarCst.clearAndCopyFrom(flattener.localVarCst);

  return success();
}

LogicalResult
getFlattenedAffineExprs(const AffineMap &map,
                        std::vector<SmallVector<int64_t, 8>> &flattenedExprs,
                        IntegerPolyhedron &cst) {
  if (map.getNumExprs() == 0) {
    cst = IntegerPolyhedron(0, 0, map.getNumDims() + map.getNumSymbols() + 1,
                            presburger::PresburgerSpace::getSetSpace(
                                map.getNumDims(), map.getNumSymbols(), 0));
    return success();
  }
  return getFlattenedAffineExprs(map.getExprs(), map.getNumDims(),
                                 map.getNumSymbols(), flattenedExprs, cst);
}

LogicalResult
getFlattenedAffineExprs(const IntegerSet &set,
                        std::vector<SmallVector<int64_t, 8>> &flattenedExprs,
                        IntegerPolyhedron &cst) {
  if (set.getNumConstraints() == 0) {
    cst = IntegerPolyhedron(0, 0, set.getNumDims() + set.getNumSymbols() + 1,
                            presburger::PresburgerSpace::getSetSpace(
                                set.getNumDims(), set.getNumSymbols(), 0));
    return success();
  }
  return getFlattenedAffineExprs(set.getConstraints(), set.getNumDims(),
                                 set.getNumSymbols(), flattenedExprs, cst);
}
} // namespace detail
} // namespace presburger
} // namespace mlir
