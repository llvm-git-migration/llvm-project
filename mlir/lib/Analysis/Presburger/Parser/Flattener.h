//===- Flattener.h - Presburger ParseStruct Flattener -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PARSER_FLATTENER_H
#define MLIR_ANALYSIS_PRESBURGER_PARSER_FLATTENER_H

#include "ParseStructs.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"

namespace mlir::presburger {
// This class is used to flatten a pure affine expression (AffineExpr,
// which is in a tree form) into a sum of products (w.r.t constants) when
// possible, and in that process simplifying the expression. For a modulo,
// floordiv, or a ceildiv expression, an additional identifier, called a local
// identifier, is introduced to rewrite the expression as a sum of product
// affine expression. Each local identifier is always and by construction a
// floordiv of a pure add/mul affine function of dimensional, symbolic, and
// other local identifiers, in a non-mutually recursive way. Hence, every local
// identifier can ultimately always be recovered as an affine function of
// dimensional and symbolic identifiers (involving floordiv's); note however
// that by AffineExpr construction, some floordiv combinations are converted to
// mod's. The result of the flattening is a flattened expression and a set of
// constraints involving just the local variables.
//
// d2 + (d0 + d1) floordiv 4  is flattened to d2 + q where 'q' is the local
// variable introduced, with localVarCst containing 4*q <= d0 + d1 <= 4*q + 3.
//
// The simplification performed includes the accumulation of contributions for
// each dimensional and symbolic identifier together, the simplification of
// floordiv/ceildiv/mod expressions and other simplifications that in turn
// happen as a result. A simplification that this flattening naturally performs
// is of simplifying the numerator and denominator of floordiv/ceildiv, and
// folding a modulo expression to a zero, if possible. Three examples are below:
//
// (d0 + 3 * d1) + d0) - 2 * d1) - d0    simplified to     d0 + d1
// (d0 - d0 mod 4 + 4) mod 4             simplified to     0
// (3*d0 + 2*d1 + d0) floordiv 2 + d1    simplified to     2*d0 + 2*d1
//
// The way the flattening works for the second example is as follows: d0 % 4 is
// replaced by d0 - 4*q with q being introduced: the expression then simplifies
// to: (d0 - (d0 - 4q) + 4) = 4q + 4, modulo of which w.r.t 4 simplifies to
// zero. Note that an affine expression may not always be expressible purely as
// a sum of products involving just the original dimensional and symbolic
// identifiers due to the presence of modulo/floordiv/ceildiv expressions that
// may not be eliminated after simplification; in such cases, the final
// expression can be reconstructed by replacing the local identifiers with their
// corresponding explicit form stored in 'localExprs' (note that each of the
// explicit forms itself would have been simplified).
//
// The expression walk method here performs a linear time post order walk that
// performs the above simplifications through visit methods, with partial
// results being stored in 'operandExprStack'. When a parent expr is visited,
// the flattened expressions corresponding to its two operands would already be
// on the stack - the parent expression looks at the two flattened expressions
// and combines the two. It pops off the operand expressions and pushes the
// combined result (although this is done in-place on its LHS operand expr).
// When the walk is completed, the flattened form of the top-level expression
// would be left on the stack.
//
// A flattener can be repeatedly used for multiple affine expressions that bind
// to the same operands, for example, for all result expressions of an
// AffineMap or AffineValueMap. In such cases, using it for multiple expressions
// is more efficient than creating a new flattener for each expression since
// common identical div and mod expressions appearing across different
// expressions are mapped to the same local identifier (same column position in
// 'localVarCst').
class AffineExprFlattener {
public:
  // Flattend expression layout: [dims, symbols, locals, constant]
  // Stack that holds the LHS and RHS operands while visiting a binary op expr.
  // In future, consider adding a prepass to determine how big the SmallVector's
  // will be, and linearize this to std::vector<int64_t> to prevent
  // SmallVector moves on re-allocation.
  std::vector<SmallVector<int64_t, 8>> operandExprStack;

  unsigned numDims;
  unsigned numSymbols;

  // Number of newly introduced identifiers to flatten mod/floordiv/ceildiv's.
  unsigned numLocals;

  // Constraints connecting newly introduced local variables (for mod's and
  // div's) to existing (dimensional and symbolic) ones. These are always
  // inequalities.
  IntegerPolyhedron localVarCst;

  // AffineExpr's corresponding to the floordiv/ceildiv/mod expressions for
  // which new identifiers were introduced; if the latter do not get canceled
  // out, these expressions can be readily used to reconstruct the AffineExpr
  // (tree) form. Note that these expressions themselves would have been
  // simplified (recursively) by this pass. Eg. d0 + (d0 + 2*d1 + d0) ceildiv 4
  // will be simplified to d0 + q, where q = (d0 + d1) ceildiv 2. (d0 + d1)
  // ceildiv 2 would be the local expression stored for q.
  SmallVector<AffineExpr, 4> localExprs;

  AffineExprFlattener(unsigned numDims, unsigned numSymbols);

  virtual ~AffineExprFlattener() = default;

  // Visitor methods.
  LogicalResult visitMulExpr(const AffineBinOpExpr &expr);
  LogicalResult visitAddExpr(const AffineBinOpExpr &expr);
  LogicalResult visitDimExpr(const AffineDimExpr &expr);
  LogicalResult visitSymbolExpr(const AffineSymbolExpr &expr);
  LogicalResult visitConstantExpr(const AffineConstantExpr &expr);
  LogicalResult visitCeilDivExpr(const AffineBinOpExpr &expr);
  LogicalResult visitFloorDivExpr(const AffineBinOpExpr &expr);

  //
  // t = expr mod c   <=>  t = expr - c*q and c*q <= expr <= c*q + c - 1
  //
  // A mod expression "expr mod c" is thus flattened by introducing a new local
  // variable q (= expr floordiv c), such that expr mod c is replaced with
  // 'expr - c * q' and c * q <= expr <= c * q + c - 1 are added to localVarCst.
  LogicalResult visitModExpr(const AffineBinOpExpr &expr);

  // Function to walk an AffineExpr (in post order).
  LogicalResult walkPostOrder(const AffineExprImpl &expr) {
    switch (expr.getKind()) {
    case AffineExprKind::Add: {
      const auto &binOpExpr = cast<AffineBinOpExpr>(expr);
      if (failed(walkOperandsPostOrder(binOpExpr)))
        return failure();
      return visitAddExpr(binOpExpr);
    }
    case AffineExprKind::Mul: {
      const auto &binOpExpr = cast<AffineBinOpExpr>(expr);
      if (failed(walkOperandsPostOrder(binOpExpr)))
        return failure();
      return visitMulExpr(binOpExpr);
    }
    case AffineExprKind::Mod: {
      const auto &binOpExpr = cast<AffineBinOpExpr>(expr);
      if (failed(walkOperandsPostOrder(binOpExpr)))
        return failure();
      return visitModExpr(binOpExpr);
    }
    case AffineExprKind::FloorDiv: {
      const auto &binOpExpr = cast<AffineBinOpExpr>(expr);
      if (failed(walkOperandsPostOrder(binOpExpr)))
        return failure();
      return visitFloorDivExpr(binOpExpr);
    }
    case AffineExprKind::CeilDiv: {
      const auto &binOpExpr = cast<AffineBinOpExpr>(expr);
      if (failed(walkOperandsPostOrder(binOpExpr)))
        return failure();
      return visitCeilDivExpr(binOpExpr);
    }
    case AffineExprKind::Constant:
      return visitConstantExpr(cast<AffineConstantExpr>(expr));
    case AffineExprKind::DimId:
      return visitDimExpr(cast<AffineDimExpr>(expr));
    case AffineExprKind::SymbolId:
      return visitSymbolExpr(cast<AffineSymbolExpr>(expr));
    }
    llvm_unreachable("Unknown AffineExpr");
  }

private:
  // Walk the operands - each operand is itself walked in post order.
  LogicalResult walkOperandsPostOrder(const AffineBinOpExpr &expr) {
    if (failed(walkPostOrder(*expr.getLHS())))
      return failure();
    if (failed(walkPostOrder(*expr.getRHS())))
      return failure();
    return success();
  }

  /// Constructs an affine expression from a flat ArrayRef. If there are local
  /// identifiers (neither dimensional nor symbolic) that appear in the sum of
  /// products expression, `localExprs` is expected to have the AffineExpr
  /// for it, and is substituted into. The ArrayRef `flatExprs` is expected to
  /// be in the format [dims, symbols, locals, constant term].
  AffineExpr getAffineExprFromFlatForm(ArrayRef<int64_t> flatExprs,
                                       unsigned numDims, unsigned numSymbols);

  // Add a local identifier (needed to flatten a mod, floordiv, ceildiv expr).
  // The local identifier added is always a floordiv of a pure add/mul affine
  // function of other identifiers, coefficients of which are specified in
  // dividend and with respect to a positive constant divisor. localExpr is the
  // simplified tree expression (AffineExpr) corresponding to the quantifier.
  void addLocalFloorDivId(ArrayRef<int64_t> dividend, int64_t divisor,
                          AffineExpr &&localExpr);

  /// Add a local identifier (needed to flatten a mod, floordiv, ceildiv, mul
  /// expr) when the rhs is a symbolic expression. The local identifier added
  /// may be a floordiv, ceildiv, mul or mod of a pure affine/semi-affine
  /// function of other identifiers, coefficients of which are specified in the
  /// lhs of the mod, floordiv, ceildiv or mul expression and with respect to a
  /// symbolic rhs expression. `localExpr` is the simplified tree expression
  /// (AffineExpr) corresponding to the quantifier.
  void addLocalIdSemiAffine(AffineExpr &&localExpr);

  /// Adds `expr`, which may be mod, ceildiv, floordiv or mod expression
  /// representing the affine expression corresponding to the quantifier
  /// introduced as the local variable corresponding to `expr`. If the
  /// quantifier is already present, we put the coefficient in the proper index
  /// of `result`, otherwise we add a new local variable and put the coefficient
  /// there.
  void addLocalVariableSemiAffine(AffineExpr &&expr,
                                  SmallVectorImpl<int64_t> &result,
                                  unsigned long resultSize);

  // t = expr floordiv c   <=> t = q, c * q <= expr <= c * q + c - 1
  // A floordiv is thus flattened by introducing a new local variable q, and
  // replacing that expression with 'q' while adding the constraints
  // c * q <= expr <= c * q + c - 1 to localVarCst (done by
  // IntegerRelation::addLocalFloorDiv).
  //
  // A ceildiv is similarly flattened:
  // t = expr ceildiv c   <=> t =  (expr + c - 1) floordiv c
  LogicalResult visitDivExpr(const AffineBinOpExpr &expr, bool isCeil);

  int findLocalId(const AffineExpr &localExpr);

  inline unsigned getNumCols() const {
    return numDims + numSymbols + numLocals + 1;
  }
  inline unsigned getConstantIndex() const { return getNumCols() - 1; }
  inline unsigned getLocalVarStartIndex() const { return numDims + numSymbols; }
  inline unsigned getSymbolStartIndex() const { return numDims; }
  inline unsigned getDimStartIndex() const { return 0; }
};

// Flattener for AffineMap.
LogicalResult
getFlattenedAffineExprs(const AffineMap &map,
                        std::vector<SmallVector<int64_t, 8>> &flattenedExprs,
                        IntegerPolyhedron &cst);

// Flattener for IntegerSet.
LogicalResult
getFlattenedAffineExprs(const IntegerSet &set,
                        std::vector<SmallVector<int64_t, 8>> &flattenedExprs,
                        IntegerPolyhedron &cst);
} // namespace mlir::presburger

#endif // MLIR_ANALYSIS_PRESBURGER_PARSER_FLATTENER_H
