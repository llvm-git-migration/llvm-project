//===- MeshOps.cpp - MLIR SPIR-V Mesh Ops  --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the mesh operations in the SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"

#include <optional>

using namespace mlir;

//===----------------------------------------------------------------------===//
// spirv.EXT.EmitMeshTasks
//===----------------------------------------------------------------------===//

LogicalResult spirv::EXTEmitMeshTasksOp::verify() {
  if (auto payloadOp = getPayload()) {
    auto payloadType = cast<spirv::PointerType>(payloadOp.getType());
    if (payloadType.getStorageClass() !=
        spirv::StorageClass::TaskPayloadWorkgroupEXT)
      return emitOpError("payload must be a variable with a storage class of "
                         "TaskPayloadWorkgroupEXT");
  }
  return success();
}
