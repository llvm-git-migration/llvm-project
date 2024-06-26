//===- SwapTransposeWithBroadcast.cpp - Swap transpose with broadcast op --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This is a pattern swap broadcast with transpose.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "linalg-swap-transpose-with-broadcast"

using namespace mlir;
using namespace mlir::linalg;

namespace {
/// This pattern canonicalize transpose by swapping the order of
/// broadcast and transpose:
///   transpose(broadcast(input)) -> broadcast(transpose(input))
struct SwapTransposeWithBroadcast : OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern<linalg::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transposeOp,
                                PatternRewriter &rewriter) const override {
    Value input = transposeOp.getInput();
    BroadcastOp broadcastOp = input.getDefiningOp<BroadcastOp>();
    if (!input.hasOneUse() || !broadcastOp)
      return failure();

    ArrayRef<int64_t> dimensions = broadcastOp.getDimensions();
    ArrayRef<int64_t> perms = transposeOp.getPermutation();

    // Get new perms and new dimensions.
    SmallVector<int64_t> resultPerms = dropDims(perms, dimensions);
    SmallVector<int64_t> invertPerm = invertPermutationVector(perms);
    SmallVector<int64_t> resultDimensions;
    for (unsigned i = 0; i < dimensions.size(); i++) {
      resultDimensions.push_back(invertPerm[dimensions[i]]);
    }

    // Create transpose result.
    Value broadcastInput = broadcastOp.getInput();
    Location loc = transposeOp.getLoc();
    MLIRContext *ctx = transposeOp.getContext();
    SmallVector<OpFoldResult> dims;
    auto broadcastInputTy =
        mlir::cast<RankedTensorType>(broadcastInput.getType());
    for (unsigned i = 0; i < broadcastInputTy.getRank(); i++) {
      if (broadcastInputTy.isDynamicDim(i)) {
        dims.push_back(rewriter.create<tensor::DimOp>(loc, broadcastInput, i)
                           ->getResult(0));
      } else {
        dims.push_back(IntegerAttr::get(IndexType::get(ctx),
                                        broadcastInputTy.getDimSize(i)));
      }
    }
    SmallVector<OpFoldResult> transposeResultShapes =
        applyPermutation(dims, resultPerms);
    Value transposeInit = rewriter.create<tensor::EmptyOp>(
        transposeOp.getLoc(), transposeResultShapes,
        broadcastInputTy.getElementType());

    // Create broadcast(transpose(input)).
    Value transposeResult =
        rewriter
            .create<TransposeOp>(loc, broadcastOp.getInput(), transposeInit,
                                 resultPerms)
            ->getResult(0);
    rewriter.replaceOpWithNewOp<BroadcastOp>(
        transposeOp, transposeResult, transposeOp.getInit(), resultDimensions);
    return success();
  }
};
} // namespace

void mlir::linalg::populateSwapTransposeWithBroadcastPatterns(
    RewritePatternSet &patterns) {
  patterns.add<SwapTransposeWithBroadcast>(patterns.getContext());
}
