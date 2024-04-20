//===- ForallToFor.cpp - scf.forall to scf.for loop conversion ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Transforms SCF.ForallOp's into SCF.ForOp's.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SCF/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_SCFFORALLTOFORLOOP
#include "mlir/Dialect/SCF/Transforms/Passes.h.inc"
} // namespace mlir

using namespace llvm;
using namespace mlir;
using scf::ForallOp;
using scf::ForOp;

LogicalResult
mlir::scf::forallToForLoop(RewriterBase &rewriter, scf::ForallOp forallOp,
                           SmallVector<Operation *> *results = nullptr) {
  rewriter.setInsertionPoint(forallOp);

  if (!forallOp.getOutputs().empty()) {
    return forallOp.emitOpError()
           << "unsupported shared outputs (didn't bufferize?)";
  }

  SmallVector<OpFoldResult> lbs = forallOp.getMixedLowerBound();
  SmallVector<OpFoldResult> ubs = forallOp.getMixedUpperBound();
  SmallVector<OpFoldResult> steps = forallOp.getMixedStep();

  auto loc = forallOp.getLoc();
  SmallVector<Value> ivs;
  for (auto &&[lb, ub, step] : llvm::zip(lbs, ubs, steps)) {
    Value lbValue = getValueOrCreateConstantIndexOp(rewriter, loc, lb);
    Value ubValue = getValueOrCreateConstantIndexOp(rewriter, loc, ub);
    Value stepValue = getValueOrCreateConstantIndexOp(rewriter, loc, step);
    auto loop =
        rewriter.create<ForOp>(loc, lbValue, ubValue, stepValue, ValueRange(),
                               [](OpBuilder &, Location, Value, ValueRange) {});
    if (results)
      results->push_back(loop);
    ivs.push_back(loop.getInductionVar());
    rewriter.setInsertionPointToStart(loop.getBody());
    rewriter.create<scf::YieldOp>(loc);
    rewriter.setInsertionPointToStart(loop.getBody());
  }
  rewriter.eraseOp(forallOp.getBody()->getTerminator());
  rewriter.inlineBlockBefore(forallOp.getBody(), &*rewriter.getInsertionPoint(),
                             ivs);
  rewriter.eraseOp(forallOp);
  return success();
}

namespace {
struct ForallToForLoopLoweringPattern : public OpRewritePattern<ForallOp> {
  using OpRewritePattern<ForallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForallOp forallOp,
                                PatternRewriter &rewriter) const override {
    if (failed(scf::forallToForLoop(rewriter, forallOp)))
      return failure();
    return success();
  }
};

struct ForallToForLoop : public impl::SCFForallToForLoopBase<ForallToForLoop> {
  void runOnOperation() override {
    auto *parentOp = getOperation();
    MLIRContext *ctx = parentOp->getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ForallToForLoopLoweringPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(parentOp, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createForallToForLoopPass() {
  return std::make_unique<ForallToForLoop>();
}
