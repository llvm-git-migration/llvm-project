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
using scf::LoopNest;

LogicalResult
mlir::scf::forallToForLoop(RewriterBase &rewriter, scf::ForallOp forallOp,
                           SmallVector<Operation *> *results = nullptr) {
  rewriter.setInsertionPoint(forallOp);

  if (!forallOp.getOutputs().empty()) {
    return forallOp.emitOpError()
           << "unsupported shared outputs (didn't bufferize?)";
  }

  auto loc = forallOp.getLoc();
  SmallVector<Value> lbs = getValueOrCreateConstantIndexOp(
      rewriter, loc, forallOp.getMixedLowerBound());
  SmallVector<Value> ubs = getValueOrCreateConstantIndexOp(
      rewriter, loc, forallOp.getMixedUpperBound());
  SmallVector<Value> steps =
      getValueOrCreateConstantIndexOp(rewriter, loc, forallOp.getMixedStep());
  LoopNest loopNest = scf::buildLoopNest(rewriter, loc, lbs, ubs, steps);

  if (results) {
    llvm::copy(loopNest.loops, std::back_inserter(*results));
  }

  SmallVector<Value> ivs;
  for (scf::ForOp loop : loopNest.loops) {
    ivs.push_back(loop.getInductionVar());
  }

  Block *innermostBlock = loopNest.loops.back().getBody();
  rewriter.eraseOp(forallOp.getBody()->getTerminator());
  rewriter.inlineBlockBefore(forallOp.getBody(), innermostBlock,
                             innermostBlock->getTerminator()->getIterator(),
                             ivs);
  rewriter.eraseOp(forallOp);

  return success();
}

namespace {
struct ForallToForLoop : public impl::SCFForallToForLoopBase<ForallToForLoop> {
  void runOnOperation() override {
    auto *parentOp = getOperation();
    IRRewriter rewriter(parentOp->getContext());

    SmallVector<scf::ForallOp> forallOps;
    parentOp->walk(
        [&](scf::ForallOp forallOp) { forallOps.push_back(forallOp); });

    for (auto forallOp : forallOps) {
      if (failed(scf::forallToForLoop(rewriter, forallOp))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace

std::unique_ptr<Pass> mlir::createForallToForLoopPass() {
  return std::make_unique<ForallToForLoop>();
}
