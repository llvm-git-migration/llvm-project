//===-- TosaTypes.td - TOSA type definitions ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the type definitions for the TOSA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TOSA_TYPES
#define TOSA_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

include "mlir/Dialect/Tosa/IR/TosaOpBase.td"

//===----------------------------------------------------------------------===//
// Tosa Type Definitions.
//===----------------------------------------------------------------------===//

// The base class for Tosa dialect types.
class Tosa_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Tosa_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

//===----------------------------------------------------------------------===//
// ShapeType
//===----------------------------------------------------------------------===//
def Tosa_Shape : Tosa_Type<"shape", "shape"> {
  let summary = "Shape with static rank and Index element type";
  let description = [{
    Syntax:

    ```
    shape-type ::= `shape` `<` rank `>`
    ```
    Values with shape type represents a shape with a fixed rank and a list of dimensions.
    Rank must be zero or a positive integer.
    Each dimension is represented by the builtin Index type.

    Examples:

    ```mlir
    // Shape with rank of four, for example, [1, 1, 8, 16]:
    !tosa.shape<4>

    // Shape with rank of one, for example, [16]:
    !tosa.shape<1>

    // Shape with rank zero, for example, [] (i.e., shape of scalar values):
    !tosa.shape<0>
    ```
  }];
  let parameters = (ins
    "int":$rank
  );
  let builders = [
   TypeBuilder<(ins "int":$rank)>
  ];
  let assemblyFormat = "`<` $rank `>`";

  let genVerifyDecl = 1;
}

def IsTosaShapeType : CPred<"mlir::tosa::isa_tosa_shape_type($_self)">;

// Whether a Tosa Shape type has a rank equal to the specified rank.
class IsTosaShapeOfRankPred<int rank> : And<[
    IsTosaShapeType,
    CPred<[{::llvm::cast<::mlir::tosa::shapeType>($_self).getRank() == }] # rank>
]>;

class TosaShapeOfRank<int rank> :
    Type<IsTosaShapeOfRankPred<rank>,
         "Tosa shape type of rank " # rank
         >;

def Rank1TosaShape : TosaShapeOfRank<1>;
def Rank2TosaShape : TosaShapeOfRank<2>;
def Rank4TosaShape : TosaShapeOfRank<4>;

#endif // TOSA_TYPES