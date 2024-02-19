//===- LLVM.cpp - C Interface for LLVM dialect ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/LLVM.h"
#include "mlir-c/IR.h"
#include "mlir-c/Support.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMOpsAttrDefs.h.inc"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include <stdint.h>

using namespace mlir;
using namespace mlir::LLVM;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(LLVM, llvm, LLVMDialect)

MlirType mlirLLVMPointerTypeGet(MlirContext ctx, unsigned addressSpace) {
  return wrap(LLVMPointerType::get(unwrap(ctx), addressSpace));
}

MlirType mlirLLVMVoidTypeGet(MlirContext ctx) {
  return wrap(LLVMVoidType::get(unwrap(ctx)));
}

MlirType mlirLLVMArrayTypeGet(MlirType elementType, unsigned numElements) {
  return wrap(LLVMArrayType::get(unwrap(elementType), numElements));
}

MlirType mlirLLVMFunctionTypeGet(MlirType resultType, intptr_t nArgumentTypes,
                                 MlirType const *argumentTypes, bool isVarArg) {
  SmallVector<Type, 2> argumentStorage;
  return wrap(LLVMFunctionType::get(
      unwrap(resultType),
      unwrapList(nArgumentTypes, argumentTypes, argumentStorage), isVarArg));
}

bool mlirTypeIsALLVMStructType(MlirType type) {
  return isa<LLVM::LLVMStructType>(unwrap(type));
}

bool mlirLLVMStructTypeIsLiteral(MlirType type) {
  return !cast<LLVM::LLVMStructType>(unwrap(type)).isIdentified();
}

intptr_t mlirLLVMStructTypeGetNumElementTypes(MlirType type) {
  return cast<LLVM::LLVMStructType>(unwrap(type)).getBody().size();
}

MlirType mlirLLVMStructTypeGetElementType(MlirType type, intptr_t position) {
  return wrap(cast<LLVM::LLVMStructType>(unwrap(type)).getBody()[position]);
}

bool mlirLLVMStructTypeIsPacked(MlirType type) {
  return cast<LLVM::LLVMStructType>(unwrap(type)).isPacked();
}

MlirStringRef mlirLLVMStructTypeGetIdentifier(MlirType type) {
  return wrap(cast<LLVM::LLVMStructType>(unwrap(type)).getName());
}

bool mlirLLVMStructTypeIsOpaque(MlirType type) {
  return cast<LLVM::LLVMStructType>(unwrap(type)).isOpaque();
}

MlirType mlirLLVMStructTypeLiteralGet(MlirContext ctx, intptr_t nFieldTypes,
                                      MlirType const *fieldTypes,
                                      bool isPacked) {
  SmallVector<Type> fieldStorage;
  return wrap(LLVMStructType::getLiteral(
      unwrap(ctx), unwrapList(nFieldTypes, fieldTypes, fieldStorage),
      isPacked));
}

MlirType mlirLLVMStructTypeLiteralGetChecked(MlirLocation loc,
                                             intptr_t nFieldTypes,
                                             MlirType const *fieldTypes,
                                             bool isPacked) {
  SmallVector<Type> fieldStorage;
  return wrap(LLVMStructType::getLiteralChecked(
      [loc]() { return emitError(unwrap(loc)); }, unwrap(loc)->getContext(),
      unwrapList(nFieldTypes, fieldTypes, fieldStorage), isPacked));
}

MlirType mlirLLVMStructTypeOpaqueGet(MlirContext ctx, MlirStringRef name) {
  return wrap(LLVMStructType::getOpaque(unwrap(name), unwrap(ctx)));
}

MlirType mlirLLVMStructTypeIdentifiedGet(MlirContext ctx, MlirStringRef name) {
  return wrap(LLVMStructType::getIdentified(unwrap(ctx), unwrap(name)));
}

MlirType mlirLLVMStructTypeIdentifiedNewGet(MlirContext ctx, MlirStringRef name,
                                            intptr_t nFieldTypes,
                                            MlirType const *fieldTypes,
                                            bool isPacked) {
  SmallVector<Type> fields;
  return wrap(LLVMStructType::getNewIdentified(
      unwrap(ctx), unwrap(name), unwrapList(nFieldTypes, fieldTypes, fields),
      isPacked));
}

MlirLogicalResult mlirLLVMStructTypeSetBody(MlirType structType,
                                            intptr_t nFieldTypes,
                                            MlirType const *fieldTypes,
                                            bool isPacked) {
  SmallVector<Type> fields;
  return wrap(
      cast<LLVM::LLVMStructType>(unwrap(structType))
          .setBody(unwrapList(nFieldTypes, fieldTypes, fields), isPacked));
}

/* fails to compile with: error: no viable conversion from 'mlir::Attribute' to
'ValueParamT' (aka 'mlir::LLVM::DIExpressionElemAttr') MlirAttribute
mlirLLVMDIExpressionAttrGet(MlirContext ctx, intptr_t nOperations, MlirAttribute
const *operations) { SmallVector<DIExpressionElemAttr, 2> valueStorage; return
wrap(DIExpressionAttr::get( unwrap(ctx), unwrapList(nOperations, operations,
valueStorage)));
}
*/

MlirAttribute mlirLLVMDINullTypeAttrGet(MlirContext ctx) {
  return wrap(DINullTypeAttr::get(unwrap(ctx)));
}

MlirAttribute mlirLLVMDIBasicTypeAttrGet(MlirContext ctx, unsigned int tag,
                                         MlirAttribute name,
                                         uint64_t sizeInBits,
                                         unsigned int encoding) {
  return wrap(DIBasicTypeAttr::get(
      unwrap(ctx), tag, cast<StringAttr>(unwrap(name)), sizeInBits, encoding));
}

/* fails to compile with: error: no viable conversion from 'mlir::Attribute' to
'ValueParamT' (aka 'mlir::LLVM::DINodeAttr') MlirAttribute
mlirLLVMDICompositeTypeAttrGet( MlirContext ctx, unsigned int tag, MlirAttribute
name, MlirAttribute file, uint32_t line, MlirAttribute scope, MlirAttribute
baseType, int64_t flags, uint64_t sizeInBits, uint64_t alignInBits, intptr_t
nElements, MlirAttribute const *elements) {
  SmallVector<DINodeAttr, 2>
elementsStorage; elementsStorage.reserve(nElements); mlir::ArrayRef<DINodeAttr>
list = unwrapList(nElements, elements, elementsStorage); return
wrap(DICompositeTypeAttr::get( unwrap(ctx), tag, cast<StringAttr>(unwrap(name)),
      cast<DIFileAttr>(unwrap(file)), line, cast<DIScopeAttr>(unwrap(scope)),
      cast<DITypeAttr>(unwrap(baseType)), DIFlags(flags), sizeInBits,
      alignInBits, list));
}
*/

MlirAttribute mlirLLVMDIDerivedTypeAttrGet(MlirContext ctx, unsigned int tag,
                                           MlirAttribute name,
                                           MlirAttribute baseType,
                                           uint64_t sizeInBits,
                                           uint32_t alignInBits,
                                           uint64_t offsetInBits) {
  return wrap(DIDerivedTypeAttr::get(unwrap(ctx), tag,
                                     cast<StringAttr>(unwrap(name)),
                                     cast<DITypeAttr>(unwrap(baseType)),
                                     sizeInBits, alignInBits, offsetInBits));
}

MlirAttribute mlirLLVMCConvAttrGet(MlirContext ctx, uint64_t cconv) {
  return wrap(CConvAttr::get(unwrap(ctx), CConv(cconv)));
}

MlirAttribute mlirLLVMComdatAttrGet(MlirContext ctx, uint64_t comdat) {
  return wrap(ComdatAttr::get(unwrap(ctx), comdat::Comdat(comdat)));
}

MlirAttribute mlirLLVMLinkageAttrGet(MlirContext ctx, uint64_t linkage) {
  return wrap(LinkageAttr::get(unwrap(ctx), linkage::Linkage(linkage)));
}

MlirAttribute mlirLLVMDIFileAttrGet(MlirContext ctx, MlirAttribute name,
                                    MlirAttribute directory) {
  return wrap(DIFileAttr::get(unwrap(ctx), cast<StringAttr>(unwrap(name)),
                              cast<StringAttr>(unwrap(directory))));
}

MlirAttribute mlirLLVMDICompileUnitAttrGet(MlirContext ctx, MlirAttribute id,
                                           unsigned int sourceLanguage,
                                           MlirAttribute file,
                                           MlirAttribute producer,
                                           bool isOptimized,
                                           uint64_t emissionKind) {
  return wrap(DICompileUnitAttr::get(
      unwrap(ctx), cast<DistinctAttr>(unwrap(id)), sourceLanguage,
      cast<DIFileAttr>(unwrap(file)), cast<StringAttr>(unwrap(producer)),
      isOptimized, DIEmissionKind(emissionKind)));
}

MlirAttribute mlirLLVMDIFlagsAttrGet(MlirContext ctx, uint64_t value) {
  return wrap(DIFlagsAttr::get(unwrap(ctx), DIFlags(value)));
}

MlirAttribute mlirLLVMDILexicalBlockAttrGet(MlirContext ctx,
                                            MlirAttribute scope,
                                            MlirAttribute file,
                                            unsigned int line,
                                            unsigned int column) {
  return wrap(
      DILexicalBlockAttr::get(unwrap(ctx), cast<DIScopeAttr>(unwrap(scope)),
                              cast<DIFileAttr>(unwrap(file)), line, column));
}

MlirAttribute mlirLLVMDILexicalBlockFileAttrGet(MlirContext ctx,
                                                MlirAttribute scope,
                                                MlirAttribute file,
                                                unsigned int discriminator) {
  return wrap(DILexicalBlockFileAttr::get(
      unwrap(ctx), cast<DIScopeAttr>(unwrap(scope)),
      cast<DIFileAttr>(unwrap(file)), discriminator));
}

MlirAttribute
mlirLLVMDILocalVariableAttrGet(MlirContext ctx, MlirAttribute scope,
                               MlirAttribute name, MlirAttribute diFile,
                               unsigned int line, unsigned int arg,
                               unsigned int alignInBits, MlirAttribute diType) {
  return wrap(DILocalVariableAttr::get(
      unwrap(ctx), cast<DIScopeAttr>(unwrap(scope)),
      cast<StringAttr>(unwrap(name)), cast<DIFileAttr>(unwrap(diFile)), line,
      arg, alignInBits, cast<DITypeAttr>(unwrap(diType))));
}

// fails to compile with error: no viable conversion from 'mlir::Attribute' to
// 'ValueParamT' (aka 'mlir::LLVM::DITypeAttr')
/*
MlirAttribute mlirLLVMDISubroutineTypeAttrGet(MlirContext ctx,
                                              unsigned int callingConvention,
                                              intptr_t nTypes,
                                              MlirAttribute const *types) {
  SmallVector<DITypeAttr, 2> elementsStorage;
  elementsStorage.reserve(nTypes);
  mlir::ArrayRef<DITypeAttr> list = unwrapList(nTypes, types, elementsStorage);
  return wrap(DISubroutineTypeAttr::get(unwrap(ctx), callingConvention, list));
}
*/

MlirAttribute mlirLLVMDISubprogramAttrGet(
    MlirContext ctx, MlirAttribute id, MlirAttribute compileUnit,
    MlirAttribute scope, MlirAttribute name, MlirAttribute linkageName,
    MlirAttribute file, unsigned int line, unsigned int scopeLine,
    uint64_t subprogramFlags, MlirAttribute type) {
  return wrap(DISubprogramAttr::get(
      unwrap(ctx), cast<DistinctAttr>(unwrap(id)),
      cast<DICompileUnitAttr>(unwrap(compileUnit)),
      cast<DIScopeAttr>(unwrap(scope)), cast<StringAttr>(unwrap(name)),
      cast<StringAttr>(unwrap(linkageName)), cast<DIFileAttr>(unwrap(file)),
      line, scopeLine, DISubprogramFlags(subprogramFlags),
      cast<DISubroutineTypeAttr>(unwrap(type))));
}

MlirAttribute mlirLLVMDIModuleAttrGet(MlirContext ctx, MlirAttribute file,
                                      MlirAttribute scope, MlirAttribute name,
                                      MlirAttribute configMacros,
                                      MlirAttribute includePath,
                                      MlirAttribute apinotes, unsigned int line,
                                      bool isDecl) {
  return wrap(DIModuleAttr::get(
      unwrap(ctx), cast<DIFileAttr>(unwrap(file)),
      cast<DIScopeAttr>(unwrap(scope)), cast<StringAttr>(unwrap(name)),
      cast<StringAttr>(unwrap(configMacros)),
      cast<StringAttr>(unwrap(includePath)), cast<StringAttr>(unwrap(apinotes)),
      line, isDecl));
}
