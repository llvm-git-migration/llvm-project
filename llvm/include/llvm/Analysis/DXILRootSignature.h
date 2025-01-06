//===- DXILRootSignature.h - DXIL Root Signature helper objects -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper objects for working with DXIL Root
/// Signatures.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_DIRECTX_HLSLROOTSIGNATURE_H
#define LLVM_DIRECTX_HLSLROOTSIGNATURE_H

#include "llvm/IR/Metadata.h"
#include "llvm/Support/ScopedPrinter.h"
namespace llvm {
namespace dxil {
namespace root_signature {

enum class RootSignatureElementKind {
  None = 0,
  RootFlags = 1,
  RootConstants = 2,
  RootDescriptor = 3,
  DescriptorTable = 4,
  StaticSampler = 5
};

enum class RootSignatureVersion {
  Version_1 = 1,
  Version_1_0 = 1,
  Version_1_1 = 2,
  Version_1_2 = 3
};

enum RootSignatureFlags : uint32_t {
  None = 0,
  AllowInputAssemblerInputLayout = 0x1,
  DenyVertexShaderRootAccess = 0x2,
  DenyHullShaderRootAccess = 0x4,
  DenyDomainShaderRootAccess = 0x8,
  DenyGeometryShaderRootAccess = 0x10,
  DenyPixelShaderRootAccess = 0x20,
  AllowStreamOutput = 0x40,
  LocalRootSignature = 0x80,
  DenyAmplificationShaderRootAccess = 0x100,
  DenyMeshShaderRootAccess = 0x200,
  CBVSRVUAVHeapDirectlyIndexed = 0x400,
  SamplerHeapDirectlyIndexed = 0x800,
  AllowLowTierReservedHwCbLimit = 0x80000000,
  ValidFlags = 0x80000fff
};

struct DxilRootSignatureDesc1_0 {
  RootSignatureFlags Flags;
};

struct VersionedRootSignatureDesc {
  RootSignatureVersion Version;
  union {
    DxilRootSignatureDesc1_0 Desc_1_0;
  };

  bool isPopulated();

  void swapBytes();
};

class MetadataParser {
public:
  NamedMDNode *Root;
  MetadataParser(NamedMDNode *Root) : Root(Root) {}

  bool Parse(RootSignatureVersion Version, VersionedRootSignatureDesc *Desc);

private:
  bool ParseRootFlags(MDNode *RootFlagRoot, VersionedRootSignatureDesc *Desc);
  bool ParseRootSignatureElement(MDNode *Element,
                                 VersionedRootSignatureDesc *Desc);
};
} // namespace root_signature
} // namespace dxil
} // namespace llvm

#endif // LLVM_DIRECTX_HLSLROOTSIGNATURE_H
