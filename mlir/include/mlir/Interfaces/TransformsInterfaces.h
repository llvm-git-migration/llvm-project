//===- TransformsInterfaces.h - Transforms interfaces -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares interfaces for managing transformations, including
// populating pattern rewrites.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_TRANSFORMSINTERFACES_H
#define MLIR_INTERFACES_TRANSFORMSINTERFACES_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
class ConversionTarget;
class RewritePatternSet;
class TypeConverter;

/// This class serves as an opaque interface for passing options to the
/// `ConversionPatternsAttrInterface` methods. Users of this class must
/// implement the `classof` method as well as using the macros
/// `MLIR_*_EXPLICIT_TYPE_ID` toensure type safeness.
class ConversionPatternAttrOptions {
public:
  ConversionPatternAttrOptions(ConversionTarget &target,
                               TypeConverter &converter);

  /// Returns the typeID.
  TypeID getTypeID() const { return typeID; }

  /// Returns a reference to the conversion target to configure.
  ConversionTarget &getConversionTarget() { return target; }

  /// Returns a reference to the type converter to configure.
  TypeConverter &getTypeConverter() { return converter; }

protected:
  /// Derived classes must use this constructor to initialize `typeID` to the
  /// appropiate value.
  ConversionPatternAttrOptions(TypeID typeID, ConversionTarget &target,
                               TypeConverter &converter);
  // Conversion target.
  ConversionTarget &target;
  // Type converter.
  TypeConverter &converter;

private:
  TypeID typeID;
};

/// Helper function for populating dialect conversion patterns. If `op`
/// implements the `OpWithTransformAttrsOpInterface` interface, then the
/// conversion pattern attributes provided by the interface will be used to
/// configure the conversion target, type converter, and the pattern set.
void populateOpConversionPatterns(Operation *op,
                                  ConversionPatternAttrOptions &options,
                                  RewritePatternSet &patterns);
} // namespace mlir

#include "mlir/Interfaces/TransformsAttrInterfaces.h.inc"

#include "mlir/Interfaces/TransformsOpInterfaces.h.inc"

MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::ConversionPatternAttrOptions)

#endif // MLIR_INTERFACES_TRANSFORMSINTERFACES_H
