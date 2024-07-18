//===- TransformsInterfaces.td - Transforms interfaces -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines interfaces for managing transformations, including populating
// pattern rewrites.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_TRANSFORMSINTERFACES_TD
#define MLIR_INTERFACES_TRANSFORMSINTERFACES_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Conversion patterns attribute interface
//===----------------------------------------------------------------------===//

def ConversionPatternsAttrInterface :
    AttrInterface<"ConversionPatternsAttrInterface"> {
  let description = [{
    This interfaces allows using attributes to configure the dialect conversion
    infrastructure, this includes:
     - The conversion target.
     - The type converter.
     - The pattern set.
    
    The conversion target and type converter are passed through the
    `ConversionPatternAttrOptions` class. Passing them through this class
    and by reference allows sub-classing the base option class, allowing
    specializations like `LLVMConversionPatternAttrOptions` for converting to
    LLVM.
  }];
  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Populate the dialect conversion target, type converter and pattern set.
      }],
      /*retTy=*/"void",
      /*methodName=*/"populateConversionPatterns",
      /*args=*/(ins "::mlir::ConversionPatternAttrOptions&":$options,
                    "::mlir::RewritePatternSet&":$patternSet)>
  ];
}

//===----------------------------------------------------------------------===//
// Operation with patterns interface
//===----------------------------------------------------------------------===//

def OpWithTransformAttrsOpInterface :
    OpInterface<"OpWithTransformAttrsOpInterface"> {
  let description = [{
    Interface for interacting with transforms attributes. These attributes
    allow configuring transformations like dialect conversion with information
    present in the IR.
  }];
  let cppNamespace = "::mlir";
  let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Populate the provided vector with a list of conversion pattern
        attributes to apply.
      }],
      /*retTy=*/"void",
      /*methodName=*/"getConversionPatternAttrs",
      /*args=*/(ins
        "::llvm::SmallVectorImpl<::mlir::ConversionPatternsAttrInterface>&":$attrs)
    >
  ];
}

#endif // MLIR_INTERFACES_TRANSFORMSINTERFACES_TD
