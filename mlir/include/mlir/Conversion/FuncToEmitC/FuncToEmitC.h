//===- FuncToEmitC.h - Func to EmitC dialect conversion ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_FUNCTOEMITC_FUNCTOEMITC_H
#define MLIR_CONVERSION_FUNCTOEMITC_FUNCTOEMITC_H

#include <memory>

namespace mlir {
class ModuleOp;
class Pass;

#define GEN_PASS_DECL_FUNCTOEMITC
#include "mlir/Conversion/Passes.h.inc"

std::unique_ptr<Pass> createConvertFuncToEmitC();

} // namespace mlir

#endif // MLIR_CONVERSION_FUNCTOEMITC_FUNCTOEMITC_H
