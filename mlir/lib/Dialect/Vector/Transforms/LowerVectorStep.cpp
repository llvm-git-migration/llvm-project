//===- LowerVectorStep.cpp - Lower 'vector.step' operation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to lower the
// 'vector.step' operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/VectorInterfaces.h"

#define DEBUG_TYPE "vector-step-lowering"

using namespace mlir;
using namespace mlir::vector;

namespace {

struct StepToArithOps : public OpRewritePattern<vector::StepOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::StepOp stepOp,
                                PatternRewriter &rewriter) const override {
    auto resultType = cast<VectorType>(stepOp.getType());
    if (!resultType.isScalable()) {
      SmallVector<APInt> indices;
      for (unsigned i = 0; i < resultType.getNumElements(); i++)
        indices.push_back(APInt(/*width=*/64, i));
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          stepOp, DenseElementsAttr::get(resultType, indices));
      return success();
    }
    return failure();
  }
};
} // namespace

void mlir::vector::populateVectorStepLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<StepToArithOps>(patterns.getContext(), benefit);
}
