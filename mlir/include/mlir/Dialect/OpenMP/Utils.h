//===- Utils.h - Utils for the OpenMP MLIR Dialect --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENMP_UTILS_H_
#define MLIR_DIALECT_OPENMP_UTILS_H_

#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::omp::utils {
mlir::ArrayAttr makeI64ArrayAttr(llvm::ArrayRef<int64_t> values,
                                 mlir::MLIRContext *context);
} // namespace mlir::omp::utils

#endif // MLIR_DIALECT_OPENMP_UTILS_H_
