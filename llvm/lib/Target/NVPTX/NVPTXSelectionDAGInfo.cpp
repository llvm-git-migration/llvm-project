//===- NVPTXSelectionDAGInfo.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NVPTXSelectionDAGInfo.h"

#define GET_SDNODE_DESC
#include "NVPTXGenSDNodeInfo.inc"

using namespace llvm;

NVPTXSelectionDAGInfo::NVPTXSelectionDAGInfo()
    : SelectionDAGGenTargetInfo(NVPTXGenSDNodeInfo) {}

NVPTXSelectionDAGInfo::~NVPTXSelectionDAGInfo() = default;

const char *NVPTXSelectionDAGInfo::getTargetNodeName(unsigned Opcode) const {
#define MAKE_CASE(V)                                                           \
  case V:                                                                      \
    return #V;

  // These nodes don't have corresponding entries in *.td files yet.
  switch (static_cast<NVPTXISD::NodeType>(Opcode)) {
    MAKE_CASE(NVPTXISD::LOAD_PARAM)
    MAKE_CASE(NVPTXISD::DeclareScalarRet)
    MAKE_CASE(NVPTXISD::CallSymbol)
    MAKE_CASE(NVPTXISD::CallSeqBegin)
    MAKE_CASE(NVPTXISD::CallSeqEnd)
    MAKE_CASE(NVPTXISD::LoadV2)
    MAKE_CASE(NVPTXISD::LoadV4)
    MAKE_CASE(NVPTXISD::LDUV2)
    MAKE_CASE(NVPTXISD::LDUV4)
    MAKE_CASE(NVPTXISD::StoreV2)
    MAKE_CASE(NVPTXISD::StoreV4)
    MAKE_CASE(NVPTXISD::SETP_F16X2)
    MAKE_CASE(NVPTXISD::SETP_BF16X2)
    MAKE_CASE(NVPTXISD::Dummy)
    MAKE_CASE(NVPTXISD::Tex1DFloatS32)
    MAKE_CASE(NVPTXISD::Tex1DFloatFloat)
    MAKE_CASE(NVPTXISD::Tex1DFloatFloatLevel)
    MAKE_CASE(NVPTXISD::Tex1DFloatFloatGrad)
    MAKE_CASE(NVPTXISD::Tex1DS32S32)
    MAKE_CASE(NVPTXISD::Tex1DS32Float)
    MAKE_CASE(NVPTXISD::Tex1DS32FloatLevel)
    MAKE_CASE(NVPTXISD::Tex1DS32FloatGrad)
    MAKE_CASE(NVPTXISD::Tex1DU32S32)
    MAKE_CASE(NVPTXISD::Tex1DU32Float)
    MAKE_CASE(NVPTXISD::Tex1DU32FloatLevel)
    MAKE_CASE(NVPTXISD::Tex1DU32FloatGrad)
    MAKE_CASE(NVPTXISD::Tex1DArrayFloatS32)
    MAKE_CASE(NVPTXISD::Tex1DArrayFloatFloat)
    MAKE_CASE(NVPTXISD::Tex1DArrayFloatFloatLevel)
    MAKE_CASE(NVPTXISD::Tex1DArrayFloatFloatGrad)
    MAKE_CASE(NVPTXISD::Tex1DArrayS32S32)
    MAKE_CASE(NVPTXISD::Tex1DArrayS32Float)
    MAKE_CASE(NVPTXISD::Tex1DArrayS32FloatLevel)
    MAKE_CASE(NVPTXISD::Tex1DArrayS32FloatGrad)
    MAKE_CASE(NVPTXISD::Tex1DArrayU32S32)
    MAKE_CASE(NVPTXISD::Tex1DArrayU32Float)
    MAKE_CASE(NVPTXISD::Tex1DArrayU32FloatLevel)
    MAKE_CASE(NVPTXISD::Tex1DArrayU32FloatGrad)
    MAKE_CASE(NVPTXISD::Tex2DFloatS32)
    MAKE_CASE(NVPTXISD::Tex2DFloatFloat)
    MAKE_CASE(NVPTXISD::Tex2DFloatFloatLevel)
    MAKE_CASE(NVPTXISD::Tex2DFloatFloatGrad)
    MAKE_CASE(NVPTXISD::Tex2DS32S32)
    MAKE_CASE(NVPTXISD::Tex2DS32Float)
    MAKE_CASE(NVPTXISD::Tex2DS32FloatLevel)
    MAKE_CASE(NVPTXISD::Tex2DS32FloatGrad)
    MAKE_CASE(NVPTXISD::Tex2DU32S32)
    MAKE_CASE(NVPTXISD::Tex2DU32Float)
    MAKE_CASE(NVPTXISD::Tex2DU32FloatLevel)
    MAKE_CASE(NVPTXISD::Tex2DU32FloatGrad)
    MAKE_CASE(NVPTXISD::Tex2DArrayFloatS32)
    MAKE_CASE(NVPTXISD::Tex2DArrayFloatFloat)
    MAKE_CASE(NVPTXISD::Tex2DArrayFloatFloatLevel)
    MAKE_CASE(NVPTXISD::Tex2DArrayFloatFloatGrad)
    MAKE_CASE(NVPTXISD::Tex2DArrayS32S32)
    MAKE_CASE(NVPTXISD::Tex2DArrayS32Float)
    MAKE_CASE(NVPTXISD::Tex2DArrayS32FloatLevel)
    MAKE_CASE(NVPTXISD::Tex2DArrayS32FloatGrad)
    MAKE_CASE(NVPTXISD::Tex2DArrayU32S32)
    MAKE_CASE(NVPTXISD::Tex2DArrayU32Float)
    MAKE_CASE(NVPTXISD::Tex2DArrayU32FloatLevel)
    MAKE_CASE(NVPTXISD::Tex2DArrayU32FloatGrad)
    MAKE_CASE(NVPTXISD::Tex3DFloatS32)
    MAKE_CASE(NVPTXISD::Tex3DFloatFloat)
    MAKE_CASE(NVPTXISD::Tex3DFloatFloatLevel)
    MAKE_CASE(NVPTXISD::Tex3DFloatFloatGrad)
    MAKE_CASE(NVPTXISD::Tex3DS32S32)
    MAKE_CASE(NVPTXISD::Tex3DS32Float)
    MAKE_CASE(NVPTXISD::Tex3DS32FloatLevel)
    MAKE_CASE(NVPTXISD::Tex3DS32FloatGrad)
    MAKE_CASE(NVPTXISD::Tex3DU32S32)
    MAKE_CASE(NVPTXISD::Tex3DU32Float)
    MAKE_CASE(NVPTXISD::Tex3DU32FloatLevel)
    MAKE_CASE(NVPTXISD::Tex3DU32FloatGrad)
    MAKE_CASE(NVPTXISD::TexCubeFloatFloat)
    MAKE_CASE(NVPTXISD::TexCubeFloatFloatLevel)
    MAKE_CASE(NVPTXISD::TexCubeS32Float)
    MAKE_CASE(NVPTXISD::TexCubeS32FloatLevel)
    MAKE_CASE(NVPTXISD::TexCubeU32Float)
    MAKE_CASE(NVPTXISD::TexCubeU32FloatLevel)
    MAKE_CASE(NVPTXISD::TexCubeArrayFloatFloat)
    MAKE_CASE(NVPTXISD::TexCubeArrayFloatFloatLevel)
    MAKE_CASE(NVPTXISD::TexCubeArrayS32Float)
    MAKE_CASE(NVPTXISD::TexCubeArrayS32FloatLevel)
    MAKE_CASE(NVPTXISD::TexCubeArrayU32Float)
    MAKE_CASE(NVPTXISD::TexCubeArrayU32FloatLevel)
    MAKE_CASE(NVPTXISD::Tld4R2DFloatFloat)
    MAKE_CASE(NVPTXISD::Tld4G2DFloatFloat)
    MAKE_CASE(NVPTXISD::Tld4B2DFloatFloat)
    MAKE_CASE(NVPTXISD::Tld4A2DFloatFloat)
    MAKE_CASE(NVPTXISD::Tld4R2DS64Float)
    MAKE_CASE(NVPTXISD::Tld4G2DS64Float)
    MAKE_CASE(NVPTXISD::Tld4B2DS64Float)
    MAKE_CASE(NVPTXISD::Tld4A2DS64Float)
    MAKE_CASE(NVPTXISD::Tld4R2DU64Float)
    MAKE_CASE(NVPTXISD::Tld4G2DU64Float)
    MAKE_CASE(NVPTXISD::Tld4B2DU64Float)
    MAKE_CASE(NVPTXISD::Tld4A2DU64Float)

    MAKE_CASE(NVPTXISD::TexUnified1DFloatS32)
    MAKE_CASE(NVPTXISD::TexUnified1DFloatFloat)
    MAKE_CASE(NVPTXISD::TexUnified1DFloatFloatLevel)
    MAKE_CASE(NVPTXISD::TexUnified1DFloatFloatGrad)
    MAKE_CASE(NVPTXISD::TexUnified1DS32S32)
    MAKE_CASE(NVPTXISD::TexUnified1DS32Float)
    MAKE_CASE(NVPTXISD::TexUnified1DS32FloatLevel)
    MAKE_CASE(NVPTXISD::TexUnified1DS32FloatGrad)
    MAKE_CASE(NVPTXISD::TexUnified1DU32S32)
    MAKE_CASE(NVPTXISD::TexUnified1DU32Float)
    MAKE_CASE(NVPTXISD::TexUnified1DU32FloatLevel)
    MAKE_CASE(NVPTXISD::TexUnified1DU32FloatGrad)
    MAKE_CASE(NVPTXISD::TexUnified1DArrayFloatS32)
    MAKE_CASE(NVPTXISD::TexUnified1DArrayFloatFloat)
    MAKE_CASE(NVPTXISD::TexUnified1DArrayFloatFloatLevel)
    MAKE_CASE(NVPTXISD::TexUnified1DArrayFloatFloatGrad)
    MAKE_CASE(NVPTXISD::TexUnified1DArrayS32S32)
    MAKE_CASE(NVPTXISD::TexUnified1DArrayS32Float)
    MAKE_CASE(NVPTXISD::TexUnified1DArrayS32FloatLevel)
    MAKE_CASE(NVPTXISD::TexUnified1DArrayS32FloatGrad)
    MAKE_CASE(NVPTXISD::TexUnified1DArrayU32S32)
    MAKE_CASE(NVPTXISD::TexUnified1DArrayU32Float)
    MAKE_CASE(NVPTXISD::TexUnified1DArrayU32FloatLevel)
    MAKE_CASE(NVPTXISD::TexUnified1DArrayU32FloatGrad)
    MAKE_CASE(NVPTXISD::TexUnified2DFloatS32)
    MAKE_CASE(NVPTXISD::TexUnified2DFloatFloat)
    MAKE_CASE(NVPTXISD::TexUnified2DFloatFloatLevel)
    MAKE_CASE(NVPTXISD::TexUnified2DFloatFloatGrad)
    MAKE_CASE(NVPTXISD::TexUnified2DS32S32)
    MAKE_CASE(NVPTXISD::TexUnified2DS32Float)
    MAKE_CASE(NVPTXISD::TexUnified2DS32FloatLevel)
    MAKE_CASE(NVPTXISD::TexUnified2DS32FloatGrad)
    MAKE_CASE(NVPTXISD::TexUnified2DU32S32)
    MAKE_CASE(NVPTXISD::TexUnified2DU32Float)
    MAKE_CASE(NVPTXISD::TexUnified2DU32FloatLevel)
    MAKE_CASE(NVPTXISD::TexUnified2DU32FloatGrad)
    MAKE_CASE(NVPTXISD::TexUnified2DArrayFloatS32)
    MAKE_CASE(NVPTXISD::TexUnified2DArrayFloatFloat)
    MAKE_CASE(NVPTXISD::TexUnified2DArrayFloatFloatLevel)
    MAKE_CASE(NVPTXISD::TexUnified2DArrayFloatFloatGrad)
    MAKE_CASE(NVPTXISD::TexUnified2DArrayS32S32)
    MAKE_CASE(NVPTXISD::TexUnified2DArrayS32Float)
    MAKE_CASE(NVPTXISD::TexUnified2DArrayS32FloatLevel)
    MAKE_CASE(NVPTXISD::TexUnified2DArrayS32FloatGrad)
    MAKE_CASE(NVPTXISD::TexUnified2DArrayU32S32)
    MAKE_CASE(NVPTXISD::TexUnified2DArrayU32Float)
    MAKE_CASE(NVPTXISD::TexUnified2DArrayU32FloatLevel)
    MAKE_CASE(NVPTXISD::TexUnified2DArrayU32FloatGrad)
    MAKE_CASE(NVPTXISD::TexUnified3DFloatS32)
    MAKE_CASE(NVPTXISD::TexUnified3DFloatFloat)
    MAKE_CASE(NVPTXISD::TexUnified3DFloatFloatLevel)
    MAKE_CASE(NVPTXISD::TexUnified3DFloatFloatGrad)
    MAKE_CASE(NVPTXISD::TexUnified3DS32S32)
    MAKE_CASE(NVPTXISD::TexUnified3DS32Float)
    MAKE_CASE(NVPTXISD::TexUnified3DS32FloatLevel)
    MAKE_CASE(NVPTXISD::TexUnified3DS32FloatGrad)
    MAKE_CASE(NVPTXISD::TexUnified3DU32S32)
    MAKE_CASE(NVPTXISD::TexUnified3DU32Float)
    MAKE_CASE(NVPTXISD::TexUnified3DU32FloatLevel)
    MAKE_CASE(NVPTXISD::TexUnified3DU32FloatGrad)
    MAKE_CASE(NVPTXISD::TexUnifiedCubeFloatFloat)
    MAKE_CASE(NVPTXISD::TexUnifiedCubeFloatFloatLevel)
    MAKE_CASE(NVPTXISD::TexUnifiedCubeS32Float)
    MAKE_CASE(NVPTXISD::TexUnifiedCubeS32FloatLevel)
    MAKE_CASE(NVPTXISD::TexUnifiedCubeU32Float)
    MAKE_CASE(NVPTXISD::TexUnifiedCubeU32FloatLevel)
    MAKE_CASE(NVPTXISD::TexUnifiedCubeArrayFloatFloat)
    MAKE_CASE(NVPTXISD::TexUnifiedCubeArrayFloatFloatLevel)
    MAKE_CASE(NVPTXISD::TexUnifiedCubeArrayS32Float)
    MAKE_CASE(NVPTXISD::TexUnifiedCubeArrayS32FloatLevel)
    MAKE_CASE(NVPTXISD::TexUnifiedCubeArrayU32Float)
    MAKE_CASE(NVPTXISD::TexUnifiedCubeArrayU32FloatLevel)
    MAKE_CASE(NVPTXISD::TexUnifiedCubeFloatFloatGrad)
    MAKE_CASE(NVPTXISD::TexUnifiedCubeS32FloatGrad)
    MAKE_CASE(NVPTXISD::TexUnifiedCubeU32FloatGrad)
    MAKE_CASE(NVPTXISD::TexUnifiedCubeArrayFloatFloatGrad)
    MAKE_CASE(NVPTXISD::TexUnifiedCubeArrayS32FloatGrad)
    MAKE_CASE(NVPTXISD::TexUnifiedCubeArrayU32FloatGrad)
    MAKE_CASE(NVPTXISD::Tld4UnifiedR2DFloatFloat)
    MAKE_CASE(NVPTXISD::Tld4UnifiedG2DFloatFloat)
    MAKE_CASE(NVPTXISD::Tld4UnifiedB2DFloatFloat)
    MAKE_CASE(NVPTXISD::Tld4UnifiedA2DFloatFloat)
    MAKE_CASE(NVPTXISD::Tld4UnifiedR2DS64Float)
    MAKE_CASE(NVPTXISD::Tld4UnifiedG2DS64Float)
    MAKE_CASE(NVPTXISD::Tld4UnifiedB2DS64Float)
    MAKE_CASE(NVPTXISD::Tld4UnifiedA2DS64Float)
    MAKE_CASE(NVPTXISD::Tld4UnifiedR2DU64Float)
    MAKE_CASE(NVPTXISD::Tld4UnifiedG2DU64Float)
    MAKE_CASE(NVPTXISD::Tld4UnifiedB2DU64Float)
    MAKE_CASE(NVPTXISD::Tld4UnifiedA2DU64Float)

    MAKE_CASE(NVPTXISD::Suld1DI8Clamp)
    MAKE_CASE(NVPTXISD::Suld1DI16Clamp)
    MAKE_CASE(NVPTXISD::Suld1DI32Clamp)
    MAKE_CASE(NVPTXISD::Suld1DI64Clamp)
    MAKE_CASE(NVPTXISD::Suld1DV2I8Clamp)
    MAKE_CASE(NVPTXISD::Suld1DV2I16Clamp)
    MAKE_CASE(NVPTXISD::Suld1DV2I32Clamp)
    MAKE_CASE(NVPTXISD::Suld1DV2I64Clamp)
    MAKE_CASE(NVPTXISD::Suld1DV4I8Clamp)
    MAKE_CASE(NVPTXISD::Suld1DV4I16Clamp)
    MAKE_CASE(NVPTXISD::Suld1DV4I32Clamp)

    MAKE_CASE(NVPTXISD::Suld1DArrayI8Clamp)
    MAKE_CASE(NVPTXISD::Suld1DArrayI16Clamp)
    MAKE_CASE(NVPTXISD::Suld1DArrayI32Clamp)
    MAKE_CASE(NVPTXISD::Suld1DArrayI64Clamp)
    MAKE_CASE(NVPTXISD::Suld1DArrayV2I8Clamp)
    MAKE_CASE(NVPTXISD::Suld1DArrayV2I16Clamp)
    MAKE_CASE(NVPTXISD::Suld1DArrayV2I32Clamp)
    MAKE_CASE(NVPTXISD::Suld1DArrayV2I64Clamp)
    MAKE_CASE(NVPTXISD::Suld1DArrayV4I8Clamp)
    MAKE_CASE(NVPTXISD::Suld1DArrayV4I16Clamp)
    MAKE_CASE(NVPTXISD::Suld1DArrayV4I32Clamp)

    MAKE_CASE(NVPTXISD::Suld2DI8Clamp)
    MAKE_CASE(NVPTXISD::Suld2DI16Clamp)
    MAKE_CASE(NVPTXISD::Suld2DI32Clamp)
    MAKE_CASE(NVPTXISD::Suld2DI64Clamp)
    MAKE_CASE(NVPTXISD::Suld2DV2I8Clamp)
    MAKE_CASE(NVPTXISD::Suld2DV2I16Clamp)
    MAKE_CASE(NVPTXISD::Suld2DV2I32Clamp)
    MAKE_CASE(NVPTXISD::Suld2DV2I64Clamp)
    MAKE_CASE(NVPTXISD::Suld2DV4I8Clamp)
    MAKE_CASE(NVPTXISD::Suld2DV4I16Clamp)
    MAKE_CASE(NVPTXISD::Suld2DV4I32Clamp)

    MAKE_CASE(NVPTXISD::Suld2DArrayI8Clamp)
    MAKE_CASE(NVPTXISD::Suld2DArrayI16Clamp)
    MAKE_CASE(NVPTXISD::Suld2DArrayI32Clamp)
    MAKE_CASE(NVPTXISD::Suld2DArrayI64Clamp)
    MAKE_CASE(NVPTXISD::Suld2DArrayV2I8Clamp)
    MAKE_CASE(NVPTXISD::Suld2DArrayV2I16Clamp)
    MAKE_CASE(NVPTXISD::Suld2DArrayV2I32Clamp)
    MAKE_CASE(NVPTXISD::Suld2DArrayV2I64Clamp)
    MAKE_CASE(NVPTXISD::Suld2DArrayV4I8Clamp)
    MAKE_CASE(NVPTXISD::Suld2DArrayV4I16Clamp)
    MAKE_CASE(NVPTXISD::Suld2DArrayV4I32Clamp)

    MAKE_CASE(NVPTXISD::Suld3DI8Clamp)
    MAKE_CASE(NVPTXISD::Suld3DI16Clamp)
    MAKE_CASE(NVPTXISD::Suld3DI32Clamp)
    MAKE_CASE(NVPTXISD::Suld3DI64Clamp)
    MAKE_CASE(NVPTXISD::Suld3DV2I8Clamp)
    MAKE_CASE(NVPTXISD::Suld3DV2I16Clamp)
    MAKE_CASE(NVPTXISD::Suld3DV2I32Clamp)
    MAKE_CASE(NVPTXISD::Suld3DV2I64Clamp)
    MAKE_CASE(NVPTXISD::Suld3DV4I8Clamp)
    MAKE_CASE(NVPTXISD::Suld3DV4I16Clamp)
    MAKE_CASE(NVPTXISD::Suld3DV4I32Clamp)

    MAKE_CASE(NVPTXISD::Suld1DI8Trap)
    MAKE_CASE(NVPTXISD::Suld1DI16Trap)
    MAKE_CASE(NVPTXISD::Suld1DI32Trap)
    MAKE_CASE(NVPTXISD::Suld1DI64Trap)
    MAKE_CASE(NVPTXISD::Suld1DV2I8Trap)
    MAKE_CASE(NVPTXISD::Suld1DV2I16Trap)
    MAKE_CASE(NVPTXISD::Suld1DV2I32Trap)
    MAKE_CASE(NVPTXISD::Suld1DV2I64Trap)
    MAKE_CASE(NVPTXISD::Suld1DV4I8Trap)
    MAKE_CASE(NVPTXISD::Suld1DV4I16Trap)
    MAKE_CASE(NVPTXISD::Suld1DV4I32Trap)

    MAKE_CASE(NVPTXISD::Suld1DArrayI8Trap)
    MAKE_CASE(NVPTXISD::Suld1DArrayI16Trap)
    MAKE_CASE(NVPTXISD::Suld1DArrayI32Trap)
    MAKE_CASE(NVPTXISD::Suld1DArrayI64Trap)
    MAKE_CASE(NVPTXISD::Suld1DArrayV2I8Trap)
    MAKE_CASE(NVPTXISD::Suld1DArrayV2I16Trap)
    MAKE_CASE(NVPTXISD::Suld1DArrayV2I32Trap)
    MAKE_CASE(NVPTXISD::Suld1DArrayV2I64Trap)
    MAKE_CASE(NVPTXISD::Suld1DArrayV4I8Trap)
    MAKE_CASE(NVPTXISD::Suld1DArrayV4I16Trap)
    MAKE_CASE(NVPTXISD::Suld1DArrayV4I32Trap)

    MAKE_CASE(NVPTXISD::Suld2DI8Trap)
    MAKE_CASE(NVPTXISD::Suld2DI16Trap)
    MAKE_CASE(NVPTXISD::Suld2DI32Trap)
    MAKE_CASE(NVPTXISD::Suld2DI64Trap)
    MAKE_CASE(NVPTXISD::Suld2DV2I8Trap)
    MAKE_CASE(NVPTXISD::Suld2DV2I16Trap)
    MAKE_CASE(NVPTXISD::Suld2DV2I32Trap)
    MAKE_CASE(NVPTXISD::Suld2DV2I64Trap)
    MAKE_CASE(NVPTXISD::Suld2DV4I8Trap)
    MAKE_CASE(NVPTXISD::Suld2DV4I16Trap)
    MAKE_CASE(NVPTXISD::Suld2DV4I32Trap)

    MAKE_CASE(NVPTXISD::Suld2DArrayI8Trap)
    MAKE_CASE(NVPTXISD::Suld2DArrayI16Trap)
    MAKE_CASE(NVPTXISD::Suld2DArrayI32Trap)
    MAKE_CASE(NVPTXISD::Suld2DArrayI64Trap)
    MAKE_CASE(NVPTXISD::Suld2DArrayV2I8Trap)
    MAKE_CASE(NVPTXISD::Suld2DArrayV2I16Trap)
    MAKE_CASE(NVPTXISD::Suld2DArrayV2I32Trap)
    MAKE_CASE(NVPTXISD::Suld2DArrayV2I64Trap)
    MAKE_CASE(NVPTXISD::Suld2DArrayV4I8Trap)
    MAKE_CASE(NVPTXISD::Suld2DArrayV4I16Trap)
    MAKE_CASE(NVPTXISD::Suld2DArrayV4I32Trap)

    MAKE_CASE(NVPTXISD::Suld3DI8Trap)
    MAKE_CASE(NVPTXISD::Suld3DI16Trap)
    MAKE_CASE(NVPTXISD::Suld3DI32Trap)
    MAKE_CASE(NVPTXISD::Suld3DI64Trap)
    MAKE_CASE(NVPTXISD::Suld3DV2I8Trap)
    MAKE_CASE(NVPTXISD::Suld3DV2I16Trap)
    MAKE_CASE(NVPTXISD::Suld3DV2I32Trap)
    MAKE_CASE(NVPTXISD::Suld3DV2I64Trap)
    MAKE_CASE(NVPTXISD::Suld3DV4I8Trap)
    MAKE_CASE(NVPTXISD::Suld3DV4I16Trap)
    MAKE_CASE(NVPTXISD::Suld3DV4I32Trap)

    MAKE_CASE(NVPTXISD::Suld1DI8Zero)
    MAKE_CASE(NVPTXISD::Suld1DI16Zero)
    MAKE_CASE(NVPTXISD::Suld1DI32Zero)
    MAKE_CASE(NVPTXISD::Suld1DI64Zero)
    MAKE_CASE(NVPTXISD::Suld1DV2I8Zero)
    MAKE_CASE(NVPTXISD::Suld1DV2I16Zero)
    MAKE_CASE(NVPTXISD::Suld1DV2I32Zero)
    MAKE_CASE(NVPTXISD::Suld1DV2I64Zero)
    MAKE_CASE(NVPTXISD::Suld1DV4I8Zero)
    MAKE_CASE(NVPTXISD::Suld1DV4I16Zero)
    MAKE_CASE(NVPTXISD::Suld1DV4I32Zero)

    MAKE_CASE(NVPTXISD::Suld1DArrayI8Zero)
    MAKE_CASE(NVPTXISD::Suld1DArrayI16Zero)
    MAKE_CASE(NVPTXISD::Suld1DArrayI32Zero)
    MAKE_CASE(NVPTXISD::Suld1DArrayI64Zero)
    MAKE_CASE(NVPTXISD::Suld1DArrayV2I8Zero)
    MAKE_CASE(NVPTXISD::Suld1DArrayV2I16Zero)
    MAKE_CASE(NVPTXISD::Suld1DArrayV2I32Zero)
    MAKE_CASE(NVPTXISD::Suld1DArrayV2I64Zero)
    MAKE_CASE(NVPTXISD::Suld1DArrayV4I8Zero)
    MAKE_CASE(NVPTXISD::Suld1DArrayV4I16Zero)
    MAKE_CASE(NVPTXISD::Suld1DArrayV4I32Zero)

    MAKE_CASE(NVPTXISD::Suld2DI8Zero)
    MAKE_CASE(NVPTXISD::Suld2DI16Zero)
    MAKE_CASE(NVPTXISD::Suld2DI32Zero)
    MAKE_CASE(NVPTXISD::Suld2DI64Zero)
    MAKE_CASE(NVPTXISD::Suld2DV2I8Zero)
    MAKE_CASE(NVPTXISD::Suld2DV2I16Zero)
    MAKE_CASE(NVPTXISD::Suld2DV2I32Zero)
    MAKE_CASE(NVPTXISD::Suld2DV2I64Zero)
    MAKE_CASE(NVPTXISD::Suld2DV4I8Zero)
    MAKE_CASE(NVPTXISD::Suld2DV4I16Zero)
    MAKE_CASE(NVPTXISD::Suld2DV4I32Zero)

    MAKE_CASE(NVPTXISD::Suld2DArrayI8Zero)
    MAKE_CASE(NVPTXISD::Suld2DArrayI16Zero)
    MAKE_CASE(NVPTXISD::Suld2DArrayI32Zero)
    MAKE_CASE(NVPTXISD::Suld2DArrayI64Zero)
    MAKE_CASE(NVPTXISD::Suld2DArrayV2I8Zero)
    MAKE_CASE(NVPTXISD::Suld2DArrayV2I16Zero)
    MAKE_CASE(NVPTXISD::Suld2DArrayV2I32Zero)
    MAKE_CASE(NVPTXISD::Suld2DArrayV2I64Zero)
    MAKE_CASE(NVPTXISD::Suld2DArrayV4I8Zero)
    MAKE_CASE(NVPTXISD::Suld2DArrayV4I16Zero)
    MAKE_CASE(NVPTXISD::Suld2DArrayV4I32Zero)

    MAKE_CASE(NVPTXISD::Suld3DI8Zero)
    MAKE_CASE(NVPTXISD::Suld3DI16Zero)
    MAKE_CASE(NVPTXISD::Suld3DI32Zero)
    MAKE_CASE(NVPTXISD::Suld3DI64Zero)
    MAKE_CASE(NVPTXISD::Suld3DV2I8Zero)
    MAKE_CASE(NVPTXISD::Suld3DV2I16Zero)
    MAKE_CASE(NVPTXISD::Suld3DV2I32Zero)
    MAKE_CASE(NVPTXISD::Suld3DV2I64Zero)
    MAKE_CASE(NVPTXISD::Suld3DV4I8Zero)
    MAKE_CASE(NVPTXISD::Suld3DV4I16Zero)
    MAKE_CASE(NVPTXISD::Suld3DV4I32Zero)
  }
#undef MAKE_CASE

  return SelectionDAGGenTargetInfo::getTargetNodeName(Opcode);
}

bool NVPTXSelectionDAGInfo::isTargetMemoryOpcode(unsigned Opcode) const {
  // These nodes don't have corresponding entries in *.td files.
  if (Opcode >= NVPTXISD::FIRST_MEMORY_OPCODE &&
      Opcode <= NVPTXISD::LAST_MEMORY_OPCODE)
    return true;

  // These nodes lack SDNPMemOperand property in *.td files.
  switch (static_cast<NVPTXISD::GenNodeType>(Opcode)) {
  default:
    break;
  case NVPTXISD::LoadParam:
  case NVPTXISD::LoadParamV2:
  case NVPTXISD::LoadParamV4:
  case NVPTXISD::StoreParam:
  case NVPTXISD::StoreParamV2:
  case NVPTXISD::StoreParamV4:
  case NVPTXISD::StoreParamS32:
  case NVPTXISD::StoreParamU32:
  case NVPTXISD::StoreRetval:
  case NVPTXISD::StoreRetvalV2:
  case NVPTXISD::StoreRetvalV4:
    return true;
  }

  return SelectionDAGGenTargetInfo::isTargetMemoryOpcode(Opcode);
}

void NVPTXSelectionDAGInfo::verifyTargetNode(const SelectionDAG &DAG,
                                             const SDNode *N) const {
  switch (N->getOpcode()) {
  default:
    break;
  case NVPTXISD::ProxyReg:
    // invalid number of results; expected 3, got 1
  case NVPTXISD::BrxEnd:
    // invalid number of results; expected 1, got 2
    return;
  }

  return SelectionDAGGenTargetInfo::verifyTargetNode(DAG, N);
}
