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
#include <memory>

namespace llvm {
namespace mcdxbc {

enum class RootSignatureElementKind {
  None = 0,
  RootFlags = 1,
  RootConstants = 2,
  RootDescriptor = 3,
  DescriptorTable = 4,
  StaticSampler = 5
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

struct RootSignatureDesc {
  uint32_t Version;
  RootSignatureFlags Flags;

  void swapBytes() {
    sys::swapByteOrder(Version);
    sys::swapByteOrder(Flags);
  }
};

class RootSignatureDescWriter {
private:
  RootSignatureDesc *Desc;

public:
  RootSignatureDescWriter(RootSignatureDesc *Desc) : Desc(Desc) {}

  void write(raw_ostream &OS,
             uint32_t Version = std::numeric_limits<uint32_t>::max());
};

} // namespace mcdxbc
} // namespace llvm

#endif // LLVM_DIRECTX_HLSLROOTSIGNATURE_H
