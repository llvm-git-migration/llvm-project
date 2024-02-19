//===-- mlir-c/Dialect/LLVM.h - C API for LLVM --------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_C_DIALECT_LLVM_H
#define MLIR_C_DIALECT_LLVM_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(LLVM, llvm);

/// Creates an llvm.ptr type.
MLIR_CAPI_EXPORTED MlirType mlirLLVMPointerTypeGet(MlirContext ctx,
                                                   unsigned addressSpace);

/// Creates an llmv.void type.
MLIR_CAPI_EXPORTED MlirType mlirLLVMVoidTypeGet(MlirContext ctx);

/// Creates an llvm.array type.
MLIR_CAPI_EXPORTED MlirType mlirLLVMArrayTypeGet(MlirType elementType,
                                                 unsigned numElements);

/// Creates an llvm.func type.
MLIR_CAPI_EXPORTED MlirType
mlirLLVMFunctionTypeGet(MlirType resultType, intptr_t nArgumentTypes,
                        MlirType const *argumentTypes, bool isVarArg);

/// Returns `true` if the type is an LLVM dialect struct type.
MLIR_CAPI_EXPORTED bool mlirTypeIsALLVMStructType(MlirType type);

/// Returns `true` if the type is a literal (unnamed) LLVM struct type.
MLIR_CAPI_EXPORTED bool mlirLLVMStructTypeIsLiteral(MlirType type);

/// Returns the number of fields in the struct. Asserts if the struct is opaque
/// or not yet initialized.
MLIR_CAPI_EXPORTED intptr_t mlirLLVMStructTypeGetNumElementTypes(MlirType type);

/// Returns the `positions`-th field of the struct. Asserts if the struct is
/// opaque, not yet initialized or if the position is out of range.
MLIR_CAPI_EXPORTED MlirType mlirLLVMStructTypeGetElementType(MlirType type,
                                                             intptr_t position);

/// Returns `true` if the struct is packed.
MLIR_CAPI_EXPORTED bool mlirLLVMStructTypeIsPacked(MlirType type);

/// Returns the identifier of the identified struct. Asserts that the struct is
/// identified, i.e., not literal.
MLIR_CAPI_EXPORTED MlirStringRef mlirLLVMStructTypeGetIdentifier(MlirType type);

/// Returns `true` is the struct is explicitly opaque (will not have a body) or
/// uninitialized (will eventually have a body).
MLIR_CAPI_EXPORTED bool mlirLLVMStructTypeIsOpaque(MlirType type);

/// Creates an LLVM literal (unnamed) struct type. This may assert if the fields
/// have types not compatible with the LLVM dialect. For a graceful failure, use
/// the checked version.
MLIR_CAPI_EXPORTED MlirType
mlirLLVMStructTypeLiteralGet(MlirContext ctx, intptr_t nFieldTypes,
                             MlirType const *fieldTypes, bool isPacked);

/// Creates an LLVM literal (unnamed) struct type if possible. Emits a
/// diagnostic at the given location and returns null otherwise.
MLIR_CAPI_EXPORTED MlirType
mlirLLVMStructTypeLiteralGetChecked(MlirLocation loc, intptr_t nFieldTypes,
                                    MlirType const *fieldTypes, bool isPacked);

/// Creates an LLVM identified struct type with no body. If a struct type with
/// this name already exists in the context, returns that type. Use
/// mlirLLVMStructTypeIdentifiedNewGet to create a fresh struct type,
/// potentially renaming it. The body should be set separatelty by calling
/// mlirLLVMStructTypeSetBody, if it isn't set already.
MLIR_CAPI_EXPORTED MlirType mlirLLVMStructTypeIdentifiedGet(MlirContext ctx,
                                                            MlirStringRef name);

/// Creates an LLVM identified struct type with no body and a name starting with
/// the given prefix. If a struct with the exact name as the given prefix
/// already exists, appends an unspecified suffix to the name so that the name
/// is unique in context.
MLIR_CAPI_EXPORTED MlirType mlirLLVMStructTypeIdentifiedNewGet(
    MlirContext ctx, MlirStringRef name, intptr_t nFieldTypes,
    MlirType const *fieldTypes, bool isPacked);

MLIR_CAPI_EXPORTED MlirType mlirLLVMStructTypeOpaqueGet(MlirContext ctx,
                                                        MlirStringRef name);

/// Sets the body of the identified struct if it hasn't been set yet. Returns
/// whether the operation was successful.
MLIR_CAPI_EXPORTED MlirLogicalResult
mlirLLVMStructTypeSetBody(MlirType structType, intptr_t nFieldTypes,
                          MlirType const *fieldTypes, bool isPacked);

/// Creates a LLVM CConv attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMCConvAttrGet(MlirContext ctx,
                                                      uint64_t cconv);

/// Creates a LLVM Comdat attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMComdatAttrGet(MlirContext ctx,
                                                       uint64_t comdat);

/// Creates a LLVM Linkage attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMLinkageAttrGet(MlirContext ctx,
                                                        uint64_t linkage);

/// Creates a LLVM DINullType attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDINullTypeAttrGet(MlirContext ctx);

/// Creates a LLVM DIExpressionElem attribute.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMDIExpressionElemAttrGet(MlirContext ctx, unsigned int opcode,
                                intptr_t nArguments, uint64_t const *arguments);

/// Creates a LLVM DIExpression attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDIExpressionAttrGet(
    MlirContext ctx, intptr_t nOperations, MlirAttribute const *operations);

/// Creates a LLVM DIBasicType attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDIBasicTypeAttrGet(
    MlirContext ctx, unsigned int tag, MlirAttribute name, uint64_t sizeInBits,
    unsigned int encoding);

/// Creates a LLVM DICompositeType attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDICompositeTypeAttrGet(
    MlirContext ctx, unsigned int tag, MlirAttribute name, MlirAttribute file,
    uint32_t line, MlirAttribute scope, MlirAttribute baseType, int64_t flags,
    uint64_t sizeInBits, uint64_t alignInBits, intptr_t nElements,
    MlirAttribute const *elements);

/// Creates a LLVM DIDerivedType attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDIDerivedTypeAttrGet(
    MlirContext ctx, unsigned int tag, MlirAttribute name,
    MlirAttribute baseType, uint64_t sizeInBits, uint32_t alignInBits,
    uint64_t offsetInBits);

/// Gets the base type from a LLVM DIDerivedType attribute.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMDIDerivedTypeAttrGetBaseType(MlirAttribute diDerivedType);

/// Creates a LLVM DIFileAttr attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDIFileAttrGet(MlirContext ctx,
                                                       MlirAttribute name,
                                                       MlirAttribute directory);

/// Creates a LLVM DICompileUnit attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDICompileUnitAttrGet(
    MlirContext ctx, MlirAttribute id, unsigned int sourceLanguage,
    MlirAttribute file, MlirAttribute producer, bool isOptimized,
    uint64_t emissionKind);

/// Creates a LLVM DIFlags attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDIFlagsAttrGet(MlirContext ctx,
                                                        uint64_t value);

/// Creates a LLVM DILexicalBlock attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDILexicalBlockAttrGet(
    MlirContext ctx, MlirAttribute scope, MlirAttribute file, unsigned int line,
    unsigned int column);

/// Creates a LLVM DILexicalBlockFile attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDILexicalBlockFileAttrGet(
    MlirContext ctx, MlirAttribute scope, MlirAttribute file,
    unsigned int discriminator);

/// Creates a LLVM DILocalVariableAttr attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDILocalVariableAttrGet(
    MlirContext ctx, MlirAttribute scope, MlirAttribute name,
    MlirAttribute diFile, unsigned int line, unsigned int arg,
    unsigned int alignInBits, MlirAttribute diType);

/// Creates a LLVM DISubprogramAttr attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDISubprogramAttrGet(
    MlirContext ctx, MlirAttribute id, MlirAttribute compileUnit,
    MlirAttribute scope, MlirAttribute name, MlirAttribute linkageName,
    MlirAttribute file, unsigned int line, unsigned int scopeLine,
    uint64_t subprogramFlags, MlirAttribute type);

/// Gets the scope from this DISubprogramAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMDISubprogramAttrGetScope(MlirAttribute diSubprogram);

/// Gets the line from this DISubprogramAttr.
MLIR_CAPI_EXPORTED unsigned int
mlirLLVMDISubprogramAttrGetLine(MlirAttribute diSubprogram);

/// Gets the scope line from this DISubprogram.
MLIR_CAPI_EXPORTED unsigned int
mlirLLVMDISubprogramAttrGetScopeLine(MlirAttribute diSubprogram);

/// Gets the compile unit from this DISubprogram.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMDISubprogramAttrGetCompileUnit(MlirAttribute diSubprogram);

/// Gets the file from this DISubprogramAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMDISubprogramAttrGetFile(MlirAttribute diSubprogram);

/// Gets the type from this DISubprogramAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMDISubprogramAttrGetType(MlirAttribute diSubprogram);

/// Creates a LLVM DISubroutineTypeAttr attribute.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMDISubroutineTypeAttrGet(MlirContext ctx, unsigned int callingConvention,
                                intptr_t nTypes, MlirAttribute const *types);

/// Creates a LLVM DIModuleAttr attribute.
MLIR_CAPI_EXPORTED MlirAttribute mlirLLVMDIModuleAttrGet(
    MlirContext ctx, MlirAttribute file, MlirAttribute scope,
    MlirAttribute name, MlirAttribute configMacros, MlirAttribute includePath,
    MlirAttribute apinotes, unsigned int line, bool isDecl);

/// Gets the scope of this DIModuleAttr.
MLIR_CAPI_EXPORTED MlirAttribute
mlirLLVMDIModuleAttrGetScope(MlirAttribute diModule);

#ifdef __cplusplus
}
#endif

#endif // MLIR_C_DIALECT_LLVM_H
