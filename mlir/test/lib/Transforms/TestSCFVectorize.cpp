//===- TestSCFVectorize.cpp - SCF vectorization test pass -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/SCFVectorize.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
struct TestSCFVectorizePass
    : public PassWrapper<TestSCFVectorizePass, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestSCFVectorizePass)

  TestSCFVectorizePass() = default;
  TestSCFVectorizePass(const TestSCFVectorizePass &pass) : PassWrapper(pass) {}

  Option<unsigned> vectorBitwidth{*this, "vector-bitwidth",
                                  llvm::cl::desc("Target vector bitwidth ")};

  StringRef getArgument() const final { return "test-scf-vectorize"; }
  StringRef getDescription() const final { return "Test SCF vectorization"; }

  virtual void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<ub::UBDialect>();
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    llvm::SmallVector<std::pair<scf::ParallelOp, SCFVectorizeParams>>
        toVectorize;

    // Simple heuristic: total number of elements processed by vector ops, but
    // prefer masked mode over non-masked.
    auto getBenefit = [](const SCFVectorizeInfo &info) {
      return info.factor * info.count * (int(info.masked) + 1);
    };

    getOperation()->walk([&](scf::ParallelOp loop) {
      std::optional<SCFVectorizeInfo> best;
      for (auto dim : llvm::seq(0u, loop.getNumLoops())) {
        auto info = getLoopVectorizeInfo(loop, dim, vectorBitwidth);
        if (!info)
          continue;

        if (!best) {
          best = *info;
          continue;
        }

        if (getBenefit(*info) > getBenefit(*best))
          best = *info;
      }

      if (!best)
        return;

      toVectorize.emplace_back(
          loop, SCFVectorizeParams{best->dim, best->factor, best->masked});
    });

    if (toVectorize.empty())
      return markAllAnalysesPreserved();

    OpBuilder builder(&getContext());
    for (auto &&[loop, params] : toVectorize) {
      builder.setInsertionPoint(loop);
      if (failed(vectorizeLoop(builder, loop, params)))
        return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace test {
void registerTesSCFVectorize() { PassRegistration<TestSCFVectorizePass>(); }
} // namespace test
} // namespace mlir
