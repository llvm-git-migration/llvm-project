//===- DeadCodeElimination.cpp - Dead Code Elimination --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Iterators.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_DEADCODEELIMINATION
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct DeadCodeElimination
    : public impl::DeadCodeEliminationBase<DeadCodeElimination> {
  void runOnOperation() override;
};
} // namespace

void DeadCodeElimination::runOnOperation() {
  Operation *topLevel = getOperation();

  // Visit operations in reverse dominance order. This visits all users before
  // their definitions. (Also takes into account unstructured control flow
  // between blocks.)
  topLevel->walk<WalkOrder::PostOrder,
                 ReverseDominanceIterator</*NoGraphRegions=*/false>>(
      [&](Operation *op) {
        // Do not remove the top-level op.
        if (op == topLevel)
          return WalkResult::advance();

        // Do not remove ops from regions that may be graph regions.
        if (mayBeGraphRegion(*op->getParentRegion()))
          return WalkResult::advance();

        // Remove dead ops.
        if (isOpTriviallyDead(op)) {
          op->erase();
          return WalkResult::skip();
        }

        return WalkResult::advance();
      });

  // ReverseDominanceIterator does not visit unreachable blocks. Erase those in
  // a second walk. First collect all reachable blocks.
  // TODO: Extend walker API to provide a callback for both ops and blocks, so
  // that reachable blocks can be collected in the same walk.
  DenseSet<Block *> reachableBlocks;
  topLevel->walk<WalkOrder::PostOrder,
                 ForwardDominanceIterator</*NoGraphRegions=*/false>>(
      [&](Block *block) { reachableBlocks.insert(block); });
  // Erase all blocks that were not visited. These are unreachable and thus
  // dead.
  topLevel->walk<WalkOrder::PostOrder>([&](Block *block) {
    if (!reachableBlocks.contains(block)) {
      block->dropAllDefinedValueUses();
      block->erase();
    }
  });
}

std::unique_ptr<Pass> mlir::createDeadCodeEliminationPass() {
  return std::make_unique<DeadCodeElimination>();
}
