//===- ConversionAttrOptions.cpp - LLVM conversion options ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines convert to LLVM options for `ConversionPatternAttr`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/ConversionAttrOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

using namespace mlir;

LLVMConversionPatternAttrOptions::LLVMConversionPatternAttrOptions(
    ConversionTarget &target, LLVMTypeConverter &converter)
    : ConversionPatternAttrOptions(
          TypeID::get<LLVMConversionPatternAttrOptions>(), target, converter) {}

LLVMTypeConverter &LLVMConversionPatternAttrOptions::getLLVMTypeConverter() {
  return static_cast<LLVMTypeConverter &>(converter);
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::LLVMConversionPatternAttrOptions)
