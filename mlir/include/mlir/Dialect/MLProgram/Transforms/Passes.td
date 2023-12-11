//===-- Passes.td - pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MLPROGRAM_TRANSFORMS_PASSES
#define MLIR_DIALECT_MLPROGRAM_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def MLProgramBufferize: Pass<"mlprogram-bufferize", "ModuleOp"> {
  let summary = "Bufferize the MLProgram dialect ops";
  let description = [{
    This pass bufferizes ops in the `ml_program` dialect. It is implemented as a
    standalone pass because One-Shot Bufferize does not handle module-level ops
    like `ml_program.global`. Therefore, this pass should run just before
    One-Shot Bufferize.

    This pass is intended to be a generic lowering of `ml_program` ops to allow
    for them to be lowered all the way to LLVM. Users may want a more specialized
    lowering depending on how they manage global state in their system.
  }];
  let dependentDialects = [
    "bufferization::BufferizationDialect", "memref::MemRefDialect", 
  ];
}

def MLProgramPipelineGlobals : Pass<"mlprogram-pipeline-globals", "ModuleOp"> {
  let summary = "Optimize `ml_program` global operations for read and store";
  let description = [{
    `ml_program`'s load and store operations can be optimized for
    write-write or write-read sets of operations. This allows known
    tensors to not be re-read when the value is already known in IR.

    The pass is designed to handle both nested regions and function calls
    safely.
  }];
  let constructor = "mlir::ml_program::createMLProgramPipelineGlobalsPass()";
}

#endif // MLIR_DIALECT_MLPROGRAM_TRANSFORMS_PASSES
