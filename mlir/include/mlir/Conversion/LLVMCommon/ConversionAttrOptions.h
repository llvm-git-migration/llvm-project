//===- ConversionAttrOptions.h - LLVM conversion options --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares convert to LLVM options for `ConversionPatternAttr`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_LLVMCOMMON_CONVERSIONATTROPTIONS_H
#define MLIR_CONVERSION_LLVMCOMMON_CONVERSIONATTROPTIONS_H

#include "mlir/Interfaces/TransformsInterfaces.h"

namespace mlir {
class LLVMTypeConverter;

/// Class for passing convert to LLVM options to `ConversionPatternAttr`
/// attributes.
class LLVMConversionPatternAttrOptions : public ConversionPatternAttrOptions {
public:
  LLVMConversionPatternAttrOptions(ConversionTarget &target,
                                   LLVMTypeConverter &converter);

  static bool classof(ConversionPatternAttrOptions const *opts) {
    return opts->getTypeID() == TypeID::get<LLVMConversionPatternAttrOptions>();
  }

  /// Get the LLVM type converter.
  LLVMTypeConverter &getLLVMTypeConverter();
};
} // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::LLVMConversionPatternAttrOptions)

#endif // MLIR_CONVERSION_LLVMCOMMON_CONVERSIONATTROPTIONS_H
