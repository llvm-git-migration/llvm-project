//===- ConvertToSPIRV.h - Conversion to SPIR-V pass ---*- C++ -*-===================//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_CONVERTTOSPIRV_CONVERTTOSPIRV_H
#define MLIR_CONVERSION_CONVERTTOSPIRV_CONVERTTOSPIRV_H

#include <memory>

#include "mlir/Pass/Pass.h"

#define GEN_PASS_DECL_CONVERTTOSPIRVPASS
#include "mlir/Conversion/Passes.h.inc"

namespace mlir {

/// Create a pass that performs dialect conversion to SPIR-V for all dialects
std::unique_ptr<OperationPass<>> createConvertToSPIRVPass();

} // namespace mlir

#endif // MLIR_CONVERSION_CONVERTTOSPIRV_CONVERTTOSPIRV_H
