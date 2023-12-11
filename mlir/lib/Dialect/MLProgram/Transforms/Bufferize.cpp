//===- Bufferize.cpp - MLProgram bufferize pass ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a bufferization pass for the MLProgram dialect
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MLProgram/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace ml_program {
#define GEN_PASS_DEF_MLPROGRAMBUFFERIZE
#include "mlir/Dialect/MLProgram/Transforms/Passes.h.inc"

static LogicalResult bufferizeMLProgramGlobalOp(GlobalOp globalOp,
                                                OpBuilder &builder) {
  if (!globalOp.getValue().has_value())
    return globalOp.emitError("global op must have a value");

  auto tensorType = cast<RankedTensorType>(globalOp.getType());
  auto memrefType =
      MemRefType::get(tensorType.getShape(), tensorType.getElementType());

  builder.setInsertionPointToStart(
      globalOp->getParentOfType<ModuleOp>().getBody());
  builder.create<memref::GlobalOp>(
      globalOp.getLoc(), globalOp.getSymName(),
      /*sym_visibility=*/globalOp.getSymVisibilityAttr(),
      /*type=*/memrefType,
      /*initial_value=*/globalOp.getValue().value(),
      /*constant=*/!globalOp.getIsMutable(),
      /*alignment=*/nullptr);
  return success();
}

static LogicalResult bufferizeMLProgramGlobalLoadOp(GlobalLoadOp globalLoadOp,
                                                    OpBuilder &builder) {
  auto loc = globalLoadOp.getLoc();
  auto tensorType = cast<RankedTensorType>(globalLoadOp.getType());
  auto memrefType =
      MemRefType::get(tensorType.getShape(), tensorType.getElementType());

  builder.setInsertionPoint(globalLoadOp);
  Value globalVal = builder.create<memref::GetGlobalOp>(
      loc, memrefType, globalLoadOp.getGlobalAttr().getLeafReference());

  // We need a copy to guarantee that the produced tensor does not alias with
  // any other buffer.
  Value alloc = builder.create<memref::AllocOp>(loc, memrefType, ValueRange{});
  builder.create<memref::CopyOp>(globalLoadOp->getLoc(), globalVal, alloc);

  globalVal = builder.create<bufferization::ToTensorOp>(loc, tensorType, alloc,
                                                        /*restrict=*/true);
  globalLoadOp->getResult(0).replaceAllUsesWith(globalVal);
  return success();
}

static LogicalResult
bufferizeMLProgramGlobalStoreOp(GlobalStoreOp globalStoreOp,
                                OpBuilder &builder) {
  auto loc = globalStoreOp.getLoc();
  auto tensorType = cast<RankedTensorType>(globalStoreOp.getValue().getType());
  auto memrefType =
      MemRefType::get(tensorType.getShape(), tensorType.getElementType());

  builder.setInsertionPoint(globalStoreOp);
  Value memref = builder.create<memref::GetGlobalOp>(
      loc, memrefType, globalStoreOp.getGlobalAttr().getLeafReference());
  Value copyValue = builder.create<bufferization::ToMemrefOp>(
      loc, memrefType, globalStoreOp.getValue());
  builder.create<memref::CopyOp>(loc, copyValue, memref);
  return success();
}

namespace {
/// Converts MLProgram operations that work on tensor-type operands or results
/// to work on buffers.
class MLProgramBufferize
    : public impl::MLProgramBufferizeBase<MLProgramBufferize> {
  void runOnOperation() override {
    auto module = getOperation();
    OpBuilder builder(module.getBodyRegion());
    SmallVector<Operation *> toErase;

    auto walkResult = module.walk([&](GlobalOp op) {
      if (auto type = dyn_cast<RankedTensorType>(op.getType())) {
        if (!type.hasStaticShape()) {
          // If the ml_program.global has dynamically shaped tensor.
          op.emitError(
              "unimplemented: global op bufferization with dynamic shape");
          return WalkResult::interrupt();
        }
      } else {
        // If the ml_program.global is of non-tensor type.
        op.emitError("unsupported global op type");
        return WalkResult::interrupt();
      }

      if (failed(bufferizeMLProgramGlobalOp(op, builder))) {
        op.emitError("bufferization for this op failed");
        return WalkResult::interrupt();
      }
      toErase.push_back(op);
      return WalkResult::advance();
    });

    if (walkResult.wasInterrupted())
      return signalPassFailure();

    module.walk([&](GlobalLoadOp op) {
      if (failed(bufferizeMLProgramGlobalLoadOp(op, builder))) {
        op.emitError("bufferization for this op failed");
        return;
      }
      toErase.push_back(op);
    });

    module.walk([&](GlobalStoreOp op) {
      if (failed(bufferizeMLProgramGlobalStoreOp(op, builder))) {
        op.emitError("bufferization for this op failed");
        return;
      }
      toErase.push_back(op);
    });

    for (auto *op : llvm::reverse(toErase))
      op->erase();
  }
};
} // namespace
} // namespace ml_program
} // namespace mlir
