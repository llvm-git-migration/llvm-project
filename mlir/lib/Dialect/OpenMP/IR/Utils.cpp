//===- Utils.cpp - Utils for the OpenMP MLIR Dialect ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenMP/Utils.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::omp::utils {
mlir::ArrayAttr makeI64ArrayAttr(llvm::ArrayRef<int64_t> values,
                                 mlir::MLIRContext *context) {
  llvm::SmallVector<mlir::Attribute, 4> attrs;
  attrs.reserve(values.size());
  for (auto &v : values)
    attrs.push_back(mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                           mlir::APInt(64, v)));
  return mlir::ArrayAttr::get(context, attrs);
}
} // namespace mlir::omp::utils
