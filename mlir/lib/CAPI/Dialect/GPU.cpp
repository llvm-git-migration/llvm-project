//===- GPU.cpp - C Interface for GPU dialect ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir-c/Dialect/GPU.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "llvm/Support/Casting.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(GPU, gpu, mlir::gpu::GPUDialect)

//===---------------------------------------------------------------------===//
// ObjectAttr
//===---------------------------------------------------------------------===//

bool mlirAttributeIsAGPUObjectAttr(MlirAttribute attr) {
  return llvm::isa<mlir::gpu::ObjectAttr>(unwrap(attr));
}

MlirAttribute mlirGPUObjectAttrGet(MlirContext mlirCtx, MlirAttribute target,
                                   uint32_t format, MlirStringRef objectStrRef,
                                   MlirAttribute mlirObjectProps) {
  mlir::MLIRContext *ctx = unwrap(mlirCtx);
  llvm::StringRef object = unwrap(objectStrRef);
  mlir::DictionaryAttr objectProps =
      llvm::cast<mlir::DictionaryAttr>(unwrap(mlirObjectProps));
  return wrap(mlir::gpu::ObjectAttr::get(
      ctx, unwrap(target), static_cast<mlir::gpu::CompilationTarget>(format),
      mlir::StringAttr::get(ctx, object), objectProps));
}

MlirAttribute mlirGPUObjectAttrGetTarget(MlirAttribute mlirObjectAttr) {
  mlir::gpu::ObjectAttr objectAttr =
      llvm::cast<mlir::gpu::ObjectAttr>(unwrap(mlirObjectAttr));
  return wrap(objectAttr.getTarget());
}

uint32_t mlirGPUObjectAttrGetFormat(MlirAttribute mlirObjectAttr) {
  mlir::gpu::ObjectAttr objectAttr =
      llvm::cast<mlir::gpu::ObjectAttr>(unwrap(mlirObjectAttr));
  return static_cast<uint32_t>(objectAttr.getFormat());
}

MlirStringRef mlirGPUObjectAttrGetObject(MlirAttribute mlirObjectAttr) {
  mlir::gpu::ObjectAttr objectAttr =
      llvm::cast<mlir::gpu::ObjectAttr>(unwrap(mlirObjectAttr));
  llvm::StringRef object = objectAttr.getObject();
  return mlirStringRefCreate(object.data(), object.size());
}

bool mlirGPUObjectAttrHasProperties(MlirAttribute mlirObjectAttr) {
  mlir::gpu::ObjectAttr objectAttr =
      llvm::cast<mlir::gpu::ObjectAttr>(unwrap(mlirObjectAttr));
  return objectAttr.getProperties() != nullptr;
}

MlirAttribute mlirGPUObjectAttrGetProperties(MlirAttribute mlirObjectAttr) {
  mlir::gpu::ObjectAttr objectAttr =
      llvm::cast<mlir::gpu::ObjectAttr>(unwrap(mlirObjectAttr));
  return wrap(objectAttr.getProperties());
}
