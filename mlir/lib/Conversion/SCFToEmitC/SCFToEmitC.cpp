//===- SCFToEmitC.cpp - SCF to EmitC conversion ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert scf.if ops into emitc ops.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
#define GEN_PASS_DEF_SCFTOEMITC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::scf;

namespace {

struct SCFToEmitCPass : public impl::SCFToEmitCBase<SCFToEmitCPass> {
  void runOnOperation() override;
};

// Lower scf::for to emitc::for, implementing result values using
// emitc::variable's updated within the loop body.
struct ForLowering : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override;
};

// Create an uninitialized emitc::variable op for each result of the given op.
template <typename T>
static SmallVector<Value> createVariablesForResults(T op,
                                                    PatternRewriter &rewriter) {
  SmallVector<Value> resultVariables;

  if (!op.getNumResults())
    return resultVariables;

  Location loc = op->getLoc();
  MLIRContext *context = op.getContext();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);

  for (OpResult result : op.getResults()) {
    Type resultType = result.getType();
    Type varType = emitc::LValueType::get(resultType);
    emitc::OpaqueAttr noInit = emitc::OpaqueAttr::get(context, "");
    emitc::VariableOp var =
        rewriter.create<emitc::VariableOp>(loc, varType, noInit);
    resultVariables.push_back(var);
  }

  return resultVariables;
}

// Create a series of assign ops assigning given values to given variables at
// the current insertion point of given rewriter.
static void assignValues(ValueRange values, SmallVector<Value> &variables,
                         PatternRewriter &rewriter, Location loc) {
  for (auto [value, var] : llvm::zip(values, variables)) {
    assert(isa<emitc::LValueType>(var.getType()) &&
           "expected var to be an lvalue type");
    assert(!isa<emitc::LValueType>(value.getType()) &&
           "expected value to not be an lvalue type");
    auto assign = rewriter.create<emitc::AssignOp>(loc, var, value);

    // TODO: Make sure this is safe, as this moves operations with memory
    // effects.
    if (auto op = dyn_cast_if_present<emitc::LValueLoadOp>(
            value.getDefiningOp())) {
      rewriter.moveOpBefore(op, assign);
    }
  }
}

static void lowerYield(SmallVector<Value> &variables, PatternRewriter &rewriter,
                       scf::YieldOp yield) {
  Location loc = yield.getLoc();
  ValueRange operands = yield.getOperands();

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(yield);

  assignValues(operands, variables, rewriter, loc);

  rewriter.create<emitc::YieldOp>(loc);
  rewriter.eraseOp(yield);
}

static void replaceUsers(PatternRewriter &rewriter,
                         SmallVector<Value> fromValues,
                         SmallVector<Value> toValues) {
  OpBuilder::InsertionGuard guard(rewriter);
  for (auto [from, to] : llvm::zip(fromValues, toValues)) {
    assert(from.getType() == cast<emitc::LValueType>(to.getType()).getValue() &&
           "expected types to match");

    for (OpOperand &operand : llvm::make_early_inc_range(from.getUses())) {
      Operation *op = operand.getOwner();
      // Skip yield ops, as these get rewritten anyways.
      if (isa<scf::YieldOp>(op)) {
        continue;
      }
      Location loc = op->getLoc();

      rewriter.setInsertionPoint(op);
      Value rValue =
          rewriter.create<emitc::LValueLoadOp>(loc, from.getType(), to);
      operand.set(rValue);
    }
  }
}

LogicalResult ForLowering::matchAndRewrite(ForOp forOp,
                                           PatternRewriter &rewriter) const {
  Location loc = forOp.getLoc();

  // Create an emitc::variable op for each result. These variables will be used
  // for the results of the operations as well as the iter_args. They are
  // assigned to by emitc::assign ops before the loop and at the end of the loop
  // body.
  SmallVector<Value> variables = createVariablesForResults(forOp, rewriter);

  // Assign initial values to the iter arg variables.
  assignValues(forOp.getInits(), variables, rewriter, loc);

  // Replace users of the iter args with variables.
  SmallVector<Value> iterArgs;
  for (BlockArgument arg : forOp.getRegionIterArgs()) {
    iterArgs.push_back(arg);
  }

  replaceUsers(rewriter, iterArgs, variables);

  emitc::ForOp loweredFor = rewriter.create<emitc::ForOp>(
      loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep());
  rewriter.eraseBlock(loweredFor.getBody());

  rewriter.inlineRegionBefore(forOp.getRegion(), loweredFor.getRegion(),
                              loweredFor.getRegion().end());
  Operation *terminator = loweredFor.getRegion().back().getTerminator();
  lowerYield(variables, rewriter, cast<scf::YieldOp>(terminator));

  // Erase block arguments for iter_args.
  loweredFor.getRegion().back().eraseArguments(1, variables.size());

  // Replace all users of the results with lazily created lvalue-to-rvalue
  // ops.
  replaceUsers(rewriter, forOp.getResults(), variables);

  rewriter.eraseOp(forOp);
  return success();
}

// Lower scf::if to emitc::if, implementing result values as emitc::variable's
// updated within the then and else regions.
struct IfLowering : public OpRewritePattern<IfOp> {
  using OpRewritePattern<IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IfOp ifOp,
                                PatternRewriter &rewriter) const override;
};

} // namespace

LogicalResult IfLowering::matchAndRewrite(IfOp ifOp,
                                          PatternRewriter &rewriter) const {
  Location loc = ifOp.getLoc();

  // Create an emitc::variable op for each result. These variables will be
  // assigned to by emitc::assign ops within the then & else regions.
  SmallVector<Value> resultVariables =
      createVariablesForResults(ifOp, rewriter);

  // Utility function to lower the contents of an scf::if region to an emitc::if
  // region. The contents of the scf::if regions is moved into the respective
  // emitc::if regions, but the scf::yield is replaced not only with an
  // emitc::yield, but also with a sequence of emitc::assign ops that set the
  // yielded values into the result variables.
  auto lowerRegion = [&resultVariables, &rewriter](Region &region,
                                                   Region &loweredRegion) {
    rewriter.inlineRegionBefore(region, loweredRegion, loweredRegion.end());
    Operation *terminator = loweredRegion.back().getTerminator();
    lowerYield(resultVariables, rewriter, cast<scf::YieldOp>(terminator));
  };

  Region &thenRegion = ifOp.getThenRegion();
  Region &elseRegion = ifOp.getElseRegion();

  bool hasElseBlock = !elseRegion.empty();

  emitc::IfOp loweredIf =
      rewriter.create<emitc::IfOp>(loc, ifOp.getCondition(), false, false);

  Region &loweredThenRegion = loweredIf.getThenRegion();
  lowerRegion(thenRegion, loweredThenRegion);

  if (hasElseBlock) {
    Region &loweredElseRegion = loweredIf.getElseRegion();
    lowerRegion(elseRegion, loweredElseRegion);
  }

  // Replace all users of the results with lazily created lvalue-to-rvalue
  // ops.
  replaceUsers(rewriter, ifOp.getResults(), resultVariables);

  rewriter.eraseOp(ifOp);
  return success();
}

void mlir::populateSCFToEmitCConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ForLowering>(patterns.getContext());
  patterns.add<IfLowering>(patterns.getContext());
}

void SCFToEmitCPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateSCFToEmitCConversionPatterns(patterns);

  // Configure conversion to lower out SCF operations.
  ConversionTarget target(getContext());
  target.addIllegalOp<scf::ForOp, scf::IfOp>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
