//===- PopulatePatternsInterfaces.h - Pattern interfaces --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces for managing transformations, including
// populating pattern rewrites.
//
//===----------------------------------------------------------------------===//

#include "mlir/Interfaces/TransformsInterfaces.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// PatternAttrsOptions
//===----------------------------------------------------------------------===//

ConversionPatternAttrOptions::ConversionPatternAttrOptions(
    ConversionTarget &target, TypeConverter &converter)
    : ConversionPatternAttrOptions(TypeID::get<ConversionPatternAttrOptions>(),
                                   target, converter) {}

ConversionPatternAttrOptions::ConversionPatternAttrOptions(
    TypeID typeID, ConversionTarget &target, TypeConverter &converter)
    : target(target), converter(converter), typeID(typeID) {}

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::ConversionPatternAttrOptions)

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

void mlir::populateOpConversionPatterns(Operation *op,
                                        ConversionPatternAttrOptions &options,
                                        RewritePatternSet &patterns) {
  auto iface = dyn_cast<OpWithTransformAttrsOpInterface>(op);
  if (!iface)
    return;
  SmallVector<ConversionPatternsAttrInterface, 12> attrs;
  iface.getConversionPatternAttrs(attrs);
  for (ConversionPatternsAttrInterface attr : attrs)
    attr.populateConversionPatterns(options, patterns);
}

#include "mlir/Interfaces/TransformsAttrInterfaces.cpp.inc"

#include "mlir/Interfaces/TransformsOpInterfaces.cpp.inc"
