//===- OuterProductFusion.cpp - Fuse 'arm_sme.outerproduct' ops -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements rewrites that fuse 'arm_sme.outerproduct' operations
// into the 2-way or 4-way widening outerproduct operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/ArmSME/Transforms/Transforms.h"
#include "mlir/Dialect/ArmSVE/IR/ArmSVEDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "arm-sme-outerproduct-fusion"

namespace mlir::arm_sme {
#define GEN_PASS_DEF_OUTERPRODUCTFUSION
#include "mlir/Dialect/ArmSME/Transforms/Passes.h.inc"
} // namespace mlir::arm_sme

using namespace mlir;
using namespace mlir::arm_sme;

namespace {
// Fuse two 'arm_sme.outerproduct' operations that are chained via the
// accumulator into 2-way outer product operation.
//
// For example:
//
//  %a0_ext = arith.extf %a0 : vector<[4]xf16> to vector<[4]xf32>
//  %b0_ext = arith.extf %b0 : vector<[4]xf16> to vector<[4]xf32>
//  %0 = arm_sme.outerproduct %a0_ext, %b0_ext : vector<[4]xf32>,
//                                               vector<[4]xf32>
//
//  %a1_ext = arith.extf %a1 : vector<[4]xf16> to vector<[4]xf32>
//  %b1_ext = arith.extf %b1 : vector<[4]xf16> to vector<[4]xf32>
//  %1 = arm_sme.outerproduct %a1_ext, %b1_ext, %0 : vector<[4]xf32>,
//                                                   vector<[4]xf32>
//
// Becomes:
//
//  %a_packed = "llvm.intr.experimental.vector.interleave2"(%a0, %a1)
//    : (vector<[4]xf16>, vector<[4]xf16>) -> vector<[8]xf16>
//  %b_packed = "llvm.intr.experimental.vector.interleave2"(%b0, %b1)
//    : (vector<[4]xf16>, vector<[4]xf16>) -> vector<[8]xf16>
//  %0 = arm_sme.fmopa_2way %a_packed, %b_packed
//    : vector<[8]xf16>, vector<[8]xf16> into vector<[4]x[4]xf32>
class OuterProductFusion2Way
    : public OpRewritePattern<arm_sme::OuterProductOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arm_sme::OuterProductOp op,
                                PatternRewriter &rewriter) const override {
    Value acc = op.getAcc();
    if (!acc)
      return rewriter.notifyMatchFailure(op, "no accumulator operand");

    arm_sme::OuterProductOp op1 = acc.getDefiningOp<arm_sme::OuterProductOp>();
    arm_sme::OuterProductOp op2 = op;
    if (!op1)
      return rewriter.notifyMatchFailure(op,
                                         "defining op of accumulator operand "
                                         "must be an 'arm_sme.outerproduct'");

    if (op1.getKind() != op2.getKind())
      return rewriter.notifyMatchFailure(
          op, "combining kind (add or sub) of outer products must match");

    if (!op1->hasOneUse()) {
      // If the first outer product has uses other than as the input to another
      // outer product, it can't be erased after fusion. This is a problem when
      // it also has an accumulator as this will be used as the root for tile
      // allocation and since the widening outer product uses the same
      // accumulator it will get assigned the same tile ID, resulting in 3
      // outer products accumulating to the same tile and incorrect results.
      //
      // Example:
      //
      //  %acc = arith.constant dense<0.0> ; root for tile allocation
      //  %0 = arm_sme.outerproduct %a0, %b0 acc(%acc)
      //  vector.print %0                  ; intermediary use, can't erase %0
      //  %1 = arm_sme.outerproduct %a1, %b1 acc(%0)
      //
      // After fusion and tile allocation
      //
      //  %0 = arm_sme.zero {tile_id = 0 : i32}
      //  %1 = arm_sme.outerproduct %a0, %b0 acc(%0) {tile_id = 0 : i32}
      //  vector.print %1
      //  %2 = arm_sme.fmopa_2way %a, %b acc(%0) {tile_id = 0 : i32}
      //
      // No accumulator would be ok, but it's simpler to prevent this
      // altogether, since it has no benefit.
      return rewriter.notifyMatchFailure(
          op, "first outer product is not single use and cannot be removed, "
              "no benefit to fusing");
    }

    if (bool(op1.getLhsMask()) != bool(op2.getLhsMask()))
      return rewriter.notifyMatchFailure(
          op, "unsupported masking, either both outerproducts are masked "
              "or neither");

    if (failed(canFuseOuterProducts(rewriter, op1, op2)))
      return failure();

    auto loc = op.getLoc();

    auto packInputs = [&](VectorType type, Value lhs, Value rhs) {
      return rewriter.create<LLVM::experimental_vector_interleave2>(loc, type,
                                                                    lhs, rhs);
    };

    auto extOp = op.getLhs().getDefiningOp();
    VectorType extSourceVectorType =
        cast<VectorType>(extOp->getOperand(0).getType());
    VectorType widenedVectorType =
        VectorType::Builder(extSourceVectorType)
            .setDim(0, extSourceVectorType.getShape()[0] * 2);
    auto lhs = packInputs(widenedVectorType,
                          op1.getLhs().getDefiningOp()->getOperand(0),
                          op2.getLhs().getDefiningOp()->getOperand(0));
    auto rhs = packInputs(widenedVectorType,
                          op1.getRhs().getDefiningOp()->getOperand(0),
                          op2.getRhs().getDefiningOp()->getOperand(0));

    Value lhsMask, rhsMask;
    if (op1.getLhsMask() || op2.getLhsMask()) {
      VectorType maskType = VectorType::Builder(widenedVectorType)
                                .setElementType(rewriter.getI1Type());
      lhsMask = packInputs(maskType, op1.getLhsMask(), op2.getLhsMask());
      rhsMask = packInputs(maskType, op1.getRhsMask(), op2.getRhsMask());
    }

    arm_sme::CombiningKind kind = op.getKind();
    if (kind == arm_sme::CombiningKind::Add) {
      TypeSwitch<Operation *>(extOp)
          .Case<arith::ExtFOp>([&](auto) {
            rewriter.replaceOpWithNewOp<arm_sme::FMopa2WayOp>(
                op2, op.getResultType(), lhs, rhs, lhsMask, rhsMask,
                op1.getAcc());
          })
          .Case<arith::ExtSIOp>([&](auto) {
            rewriter.replaceOpWithNewOp<arm_sme::SMopa2WayOp>(
                op2, op.getResultType(), lhs, rhs, lhsMask, rhsMask,
                op1.getAcc());
          })
          .Case<arith::ExtUIOp>([&](auto) {
            rewriter.replaceOpWithNewOp<arm_sme::UMopa2WayOp>(
                op2, op.getResultType(), lhs, rhs, lhsMask, rhsMask,
                op1.getAcc());
          })
          .Default([&](auto) { llvm_unreachable("unexpected extend op!"); });
    } else if (kind == arm_sme::CombiningKind::Sub) {
      TypeSwitch<Operation *>(extOp)
          .Case<arith::ExtFOp>([&](auto) {
            rewriter.replaceOpWithNewOp<arm_sme::FMops2WayOp>(
                op2, op.getResultType(), lhs, rhs, lhsMask, rhsMask,
                op1.getAcc());
          })
          .Case<arith::ExtSIOp>([&](auto) {
            rewriter.replaceOpWithNewOp<arm_sme::SMops2WayOp>(
                op2, op.getResultType(), lhs, rhs, lhsMask, rhsMask,
                op1.getAcc());
          })
          .Case<arith::ExtUIOp>([&](auto) {
            rewriter.replaceOpWithNewOp<arm_sme::UMops2WayOp>(
                op2, op.getResultType(), lhs, rhs, lhsMask, rhsMask,
                op1.getAcc());
          })
          .Default([&](auto) { llvm_unreachable("unexpected extend op!"); });
    } else {
      llvm_unreachable("unexpected arm_sme::CombiningKind!");
    }

    rewriter.eraseOp(op1);

    return success();
  }

private:
  // A pair of outer product can be fused if all of the following are true:
  // - input and result types match.
  // - the defining operations of the inputs are identical extensions,
  //   specifically either:
  //     - a signed or unsigned extension for integer types.
  //     - a floating-point extension for floating-point types.
  // - the types and extension are supported, i.e. there's a 2-way operation
  //   they can be fused into.
  LogicalResult canFuseOuterProducts(PatternRewriter &rewriter,
                                     arm_sme::OuterProductOp op1,
                                     arm_sme::OuterProductOp op2) const {
    // Supported result types.
    auto nxnxv4i32 =
        VectorType::get({4, 4}, rewriter.getI32Type(), {true, true});
    auto nxnxv4f32 =
        VectorType::get({4, 4}, rewriter.getF32Type(), {true, true});
    // Supported input types.
    // Note: this is before packing so these have half the number of elements
    // of the input vector types of the 2-way operations.
    auto nxv4i16 = VectorType::get({4}, rewriter.getI16Type(), true);
    auto nxv4f16 = VectorType::get({4}, rewriter.getF16Type(), true);
    auto nxv4bf16 = VectorType::get({4}, rewriter.getBF16Type(), true);
    if ((failed(
             isCompatible<arith::ExtFOp>(rewriter, op1, nxnxv4f32, nxv4f16)) ||
         failed(
             isCompatible<arith::ExtFOp>(rewriter, op2, nxnxv4f32, nxv4f16))) &&
        (failed(
             isCompatible<arith::ExtFOp>(rewriter, op1, nxnxv4f32, nxv4bf16)) ||
         failed(isCompatible<arith::ExtFOp>(rewriter, op2, nxnxv4f32,
                                            nxv4bf16))) &&
        (failed(
             isCompatible<arith::ExtSIOp>(rewriter, op1, nxnxv4i32, nxv4i16)) ||
         failed(isCompatible<arith::ExtSIOp>(rewriter, op2, nxnxv4i32,
                                             nxv4i16))) &&
        (failed(
             isCompatible<arith::ExtUIOp>(rewriter, op1, nxnxv4i32, nxv4i16)) ||
         failed(
             isCompatible<arith::ExtUIOp>(rewriter, op2, nxnxv4i32, nxv4i16))))
      return failure();

    return success();
  }

  // An outer product is compatible if all of the following are true:
  // - the result type matches `resultType`.
  // - the defining operations of the inputs are identical and of the type
  //   `ExtOp`.
  // - the input types of the defining operations are identical and match
  //   `inputType`.
  template <typename ExtOp>
  LogicalResult isCompatible(PatternRewriter &rewriter,
                             arm_sme::OuterProductOp op, VectorType resultType,
                             VectorType inputType) const {
    if (op.getResultType() != resultType)
      return rewriter.notifyMatchFailure(op.getLoc(), [&](Diagnostic &diag) {
        diag << "unsupported result type, expected " << resultType;
      });

    auto lhsDefOp = op.getLhs().getDefiningOp<ExtOp>();
    auto rhsDefOp = op.getRhs().getDefiningOp<ExtOp>();

    if (!lhsDefOp || !rhsDefOp)
      return rewriter.notifyMatchFailure(
          op, "defining op of outerproduct operands must be 'arith.extf' or "
              "'arith.extsi' or 'arith.extui'");

    auto lhsInType = cast<VectorType>(lhsDefOp->getOperand(0).getType());
    auto rhsInType = cast<VectorType>(rhsDefOp->getOperand(0).getType());

    if (lhsInType != inputType || rhsInType != inputType)
      return rewriter.notifyMatchFailure(op.getLoc(), [&](Diagnostic &diag) {
        diag << "unsupported input type, expected " << inputType;
      });

    return success();
  }
};

// Fold four 'arm_sme.outerproduct' operations that are chained via the
// accumulator into 4-way outer product operation.
class OuterProductFusion4Way
    : public OpRewritePattern<arm_sme::OuterProductOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arm_sme::OuterProductOp op,
                                PatternRewriter &rewriter) const override {
    Value acc = op.getAcc();
    if (!acc)
      return rewriter.notifyMatchFailure(op, "no accumulator operand");

    arm_sme::OuterProductOp op4 = op;
    arm_sme::OuterProductOp op3 = acc.getDefiningOp<arm_sme::OuterProductOp>();
    if (!op3)
      return rewriter.notifyMatchFailure(op,
                                         "defining op of accumulator operand "
                                         "must be an 'arm_sme.outerproduct'");

    acc = op3.getAcc();
    if (!acc)
      return rewriter.notifyMatchFailure(op, "no accumulator operand");

    arm_sme::OuterProductOp op2 = acc.getDefiningOp<arm_sme::OuterProductOp>();
    if (!op2)
      return rewriter.notifyMatchFailure(op,
                                         "defining op of accumulator operand "
                                         "must be an 'arm_sme.outerproduct'");

    acc = op2.getAcc();
    if (!acc)
      return rewriter.notifyMatchFailure(op, "no accumulator operand");

    arm_sme::OuterProductOp op1 = acc.getDefiningOp<arm_sme::OuterProductOp>();
    if (!op1)
      return rewriter.notifyMatchFailure(op,
                                         "defining op of accumulator operand "
                                         "must be an 'arm_sme.outerproduct'");

    arm_sme::CombiningKind kind = op1.getKind();
    if (op2.getKind() != kind || op3.getKind() != kind || op4.getKind() != kind)
      return rewriter.notifyMatchFailure(
          op, "combining kind (add or sub) of outer products must match");

    if (!llvm::hasSingleElement(op1->getUses()) ||
        !llvm::hasSingleElement(op2->getUses()) ||
        !llvm::hasSingleElement(op3->getUses()))
      return rewriter.notifyMatchFailure(
          op, "outer products are not single use and cannot be removed, "
              "no benefit to widening");

    auto nxnxv4i32 =
        VectorType::get({4, 4}, rewriter.getI32Type(), {true, true});
    auto nxnxv2i64 =
        VectorType::get({2, 2}, rewriter.getI64Type(), {true, true});
    auto nxv4i8 = VectorType::get({4}, rewriter.getI8Type(), true);
    auto nxv2i16 = VectorType::get({2}, rewriter.getI16Type(), true);
    if (
        // signed, i8i8i32
        (failed(
             isWidenable<arith::ExtSIOp>(rewriter, op1, nxnxv4i32, nxv4i8)) ||
         failed(
             isWidenable<arith::ExtSIOp>(rewriter, op2, nxnxv4i32, nxv4i8)) ||
         failed(
             isWidenable<arith::ExtSIOp>(rewriter, op3, nxnxv4i32, nxv4i8)) ||
         failed(
             isWidenable<arith::ExtSIOp>(rewriter, op4, nxnxv4i32, nxv4i8))) &&
        // signed, i16i16i64
        (failed(
             isWidenable<arith::ExtSIOp>(rewriter, op1, nxnxv2i64, nxv2i16)) ||
         failed(
             isWidenable<arith::ExtSIOp>(rewriter, op2, nxnxv2i64, nxv2i16)) ||
         failed(
             isWidenable<arith::ExtSIOp>(rewriter, op3, nxnxv2i64, nxv2i16)) ||
         failed(
             isWidenable<arith::ExtSIOp>(rewriter, op4, nxnxv2i64, nxv2i16))) &&
        // unsigned, i8i8i32
        (failed(
             isWidenable<arith::ExtUIOp>(rewriter, op1, nxnxv4i32, nxv4i8)) ||
         failed(
             isWidenable<arith::ExtUIOp>(rewriter, op2, nxnxv4i32, nxv4i8)) ||
         failed(
             isWidenable<arith::ExtUIOp>(rewriter, op3, nxnxv4i32, nxv4i8)) ||
         failed(
             isWidenable<arith::ExtUIOp>(rewriter, op4, nxnxv4i32, nxv4i8))) &&
        // unsigned, i16i16i64
        (failed(
             isWidenable<arith::ExtUIOp>(rewriter, op1, nxnxv2i64, nxv2i16)) ||
         failed(
             isWidenable<arith::ExtUIOp>(rewriter, op2, nxnxv2i64, nxv2i16)) ||
         failed(
             isWidenable<arith::ExtUIOp>(rewriter, op3, nxnxv2i64, nxv2i16)) ||
         failed(
             isWidenable<arith::ExtUIOp>(rewriter, op4, nxnxv2i64, nxv2i16))) &&
        // signed by unsigned, i8i8i32
        (failed(isWidenable<arith::ExtSIOp, arith::ExtUIOp>(
             rewriter, op1, nxnxv4i32, nxv4i8)) ||
         failed(isWidenable<arith::ExtSIOp, arith::ExtUIOp>(
             rewriter, op2, nxnxv4i32, nxv4i8)) ||
         failed(isWidenable<arith::ExtSIOp, arith::ExtUIOp>(
             rewriter, op3, nxnxv4i32, nxv4i8)) ||
         failed(isWidenable<arith::ExtSIOp, arith::ExtUIOp>(
             rewriter, op4, nxnxv4i32, nxv4i8))) &&
        // signed by unsigned, i16i16i64
        (failed(isWidenable<arith::ExtSIOp, arith::ExtUIOp>(
             rewriter, op1, nxnxv2i64, nxv2i16)) ||
         failed(isWidenable<arith::ExtSIOp, arith::ExtUIOp>(
             rewriter, op2, nxnxv2i64, nxv2i16)) ||
         failed(isWidenable<arith::ExtSIOp, arith::ExtUIOp>(
             rewriter, op3, nxnxv2i64, nxv2i16)) ||
         failed(isWidenable<arith::ExtSIOp, arith::ExtUIOp>(
             rewriter, op4, nxnxv2i64, nxv2i16))) &&
        // unsigned by signed, i8i8i32
        (failed(isWidenable<arith::ExtUIOp, arith::ExtSIOp>(
             rewriter, op1, nxnxv4i32, nxv4i8)) ||
         failed(isWidenable<arith::ExtUIOp, arith::ExtSIOp>(
             rewriter, op2, nxnxv4i32, nxv4i8)) ||
         failed(isWidenable<arith::ExtUIOp, arith::ExtSIOp>(
             rewriter, op3, nxnxv4i32, nxv4i8)) ||
         failed(isWidenable<arith::ExtUIOp, arith::ExtSIOp>(
             rewriter, op4, nxnxv4i32, nxv4i8))) &&
        // unsigned by signed, i16i16i64
        (failed(isWidenable<arith::ExtUIOp, arith::ExtSIOp>(
             rewriter, op1, nxnxv2i64, nxv2i16)) ||
         failed(isWidenable<arith::ExtUIOp, arith::ExtSIOp>(
             rewriter, op2, nxnxv2i64, nxv2i16)) ||
         failed(isWidenable<arith::ExtUIOp, arith::ExtSIOp>(
             rewriter, op3, nxnxv2i64, nxv2i16)) ||
         failed(isWidenable<arith::ExtUIOp, arith::ExtSIOp>(
             rewriter, op4, nxnxv2i64, nxv2i16))))
      return failure();

    auto loc = op.getLoc();

    auto packInputs = [&](Value lhs, Value rhs) {
      auto inputType = cast<VectorType>(lhs.getType());
      VectorType widenedType =
          VectorType::Builder(inputType).setDim(0, inputType.getShape()[0] * 2);
      return rewriter.create<LLVM::experimental_vector_interleave2>(
          loc, widenedType, lhs, rhs);
    };

    auto lhsExtOp = op.getLhs().getDefiningOp();
    auto rhsExtOp = op.getRhs().getDefiningOp();
    auto lhs0 = packInputs(op1.getLhs().getDefiningOp()->getOperand(0),
                           op3.getLhs().getDefiningOp()->getOperand(0));
    auto lhs1 = packInputs(op2.getLhs().getDefiningOp()->getOperand(0),
                           op4.getLhs().getDefiningOp()->getOperand(0));
    auto lhs = packInputs(lhs0, lhs1);

    auto rhs0 = packInputs(op1.getRhs().getDefiningOp()->getOperand(0),
                           op3.getRhs().getDefiningOp()->getOperand(0));
    auto rhs1 = packInputs(op2.getRhs().getDefiningOp()->getOperand(0),
                           op4.getRhs().getDefiningOp()->getOperand(0));
    auto rhs = packInputs(rhs0, rhs1);

    Value lhsMask, rhsMask;
    if (op1.getLhsMask() || op2.getLhsMask() || op3.getLhsMask() ||
        op4.getLhsMask()) {
      if (!(op1.getLhsMask() && op2.getLhsMask() && op3.getLhsMask() &&
            op4.getLhsMask()))
        return rewriter.notifyMatchFailure(
            op, "unsupported masking, either all outerproducts are masked "
                "or none");

      auto lhs0Mask = packInputs(op1.getLhsMask(), op3.getLhsMask());
      auto lhs1Mask = packInputs(op2.getLhsMask(), op4.getLhsMask());
      lhsMask = packInputs(lhs0Mask, lhs1Mask);

      auto rhs0Mask = packInputs(op1.getRhsMask(), op3.getRhsMask());
      auto rhs1Mask = packInputs(op2.getRhsMask(), op4.getRhsMask());
      rhsMask = packInputs(rhs0Mask, rhs1Mask);
    }

    assert((kind == arm_sme::CombiningKind::Add ||
            kind == arm_sme::CombiningKind::Sub) &&
           "unhandled arm_sme::CombiningKind!");
    if (isa<arith::ExtSIOp>(lhsExtOp) && isa<arith::ExtSIOp>(rhsExtOp)) {
      if (kind == arm_sme::CombiningKind::Add)
        rewriter.replaceOpWithNewOp<arm_sme::SMopa4WayOp>(
            op4, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
      else
        rewriter.replaceOpWithNewOp<arm_sme::SMops4WayOp>(
            op4, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
    } else if (isa<arith::ExtUIOp>(lhsExtOp) && isa<arith::ExtUIOp>(rhsExtOp)) {
      if (kind == arm_sme::CombiningKind::Add)
        rewriter.replaceOpWithNewOp<arm_sme::UMopa4WayOp>(
            op4, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
      else
        rewriter.replaceOpWithNewOp<arm_sme::UMops4WayOp>(
            op4, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
    } else if (isa<arith::ExtSIOp>(lhsExtOp) && isa<arith::ExtUIOp>(rhsExtOp)) {
      if (kind == arm_sme::CombiningKind::Add)
        rewriter.replaceOpWithNewOp<arm_sme::SuMopa4WayOp>(
            op4, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
      else
        rewriter.replaceOpWithNewOp<arm_sme::SuMops4WayOp>(
            op4, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
    } else if (isa<arith::ExtUIOp>(lhsExtOp) && isa<arith::ExtSIOp>(rhsExtOp)) {
      if (kind == arm_sme::CombiningKind::Add)
        rewriter.replaceOpWithNewOp<arm_sme::UsMopa4WayOp>(
            op4, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
      else
        rewriter.replaceOpWithNewOp<arm_sme::UsMops4WayOp>(
            op4, op.getResultType(), lhs, rhs, lhsMask, rhsMask, op1.getAcc());
    } else
      llvm_unreachable("unexpected extend op!");

    op3.erase();
    op2.erase();
    op1.erase();

    return success();
  }

private:
  template <typename LhsExtOp, typename RhsExtOp = LhsExtOp>
  LogicalResult isWidenable(PatternRewriter &rewriter,
                            arm_sme::OuterProductOp op, VectorType resultType,
                            VectorType inputType) const {
    if (op.getResultType() != resultType)
      return rewriter.notifyMatchFailure(
          op, "unsupported result type, expected 'vector<[4]x[4]xi32>' or "
              "'vector<[2]x[2]xi64>'");

    auto lhsDefOp = op.getLhs().getDefiningOp<LhsExtOp>();
    auto rhsDefOp = op.getRhs().getDefiningOp<RhsExtOp>();

    if (!lhsDefOp || !rhsDefOp)
      return rewriter.notifyMatchFailure(
          op, "defining op of outerproduct operands must be 'arith.extsi' or "
              "'arith.extui'");

    auto lhsInType = cast<VectorType>(lhsDefOp->getOperand(0).getType());
    auto rhsInType = cast<VectorType>(rhsDefOp->getOperand(0).getType());

    if (lhsInType != inputType || rhsInType != inputType)
      return rewriter.notifyMatchFailure(
          op, "unsupported input types, expected 'vector<[4]xi8>' or "
              "'vector<[2]xi16>'");
    return success();
  }
};

struct OuterProductFusionPass
    : public arm_sme::impl::OuterProductFusionBase<OuterProductFusionPass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateOuterProductFusionPatterns(patterns);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

void mlir::arm_sme::populateOuterProductFusionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<OuterProductFusion2Way, OuterProductFusion4Way>(
      patterns.getContext());
}

std::unique_ptr<Pass> mlir::arm_sme::createOuterProductFusionPass() {
  return std::make_unique<OuterProductFusionPass>();
}
