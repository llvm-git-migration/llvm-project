//===- DXILRootSignature.cpp - DXIL Root Signature helper objects
//-----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains the parsing logic to extract root signature data
///       from LLVM IR metadata.
///
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DXILRootSignature.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>

namespace llvm {
namespace dxil {

bool root_signature::MetadataParser::Parse(RootSignatureVersion Version,
                                           VersionedRootSignatureDesc *Desc) {
  Desc->Version = Version;
  bool HasError = false;

  for (unsigned int Sid = 0; Sid < Root->getNumOperands(); Sid++) {
    // This should be an if, for error handling
    MDNode *Node = cast<MDNode>(Root->getOperand(Sid));

    // Not sure what use this for...
    Metadata *Func = Node->getOperand(0).get();

    // This should be an if, for error handling
    MDNode *Elements = cast<MDNode>(Node->getOperand(1).get());

    for (unsigned int Eid = 0; Eid < Elements->getNumOperands(); Eid++) {
      MDNode *Element = cast<MDNode>(Elements->getOperand(Eid));

      HasError = HasError || ParseRootSignatureElement(Element, Desc);
    }
  }
  return HasError;
}

bool root_signature::MetadataParser::ParseRootFlags(
    MDNode *RootFlagNode, VersionedRootSignatureDesc *Desc) {

  assert(RootFlagNode->getNumOperands() == 2 &&
         "Invalid format for RootFlag Element");
  auto *Flag = mdconst::extract<ConstantInt>(RootFlagNode->getOperand(1));
  auto Value = (RootSignatureFlags)Flag->getZExtValue();

  if ((Value & ~RootSignatureFlags::ValidFlags) != RootSignatureFlags::None)
    return true;

  switch (Desc->Version) {

  case RootSignatureVersion::Version_1:
    Desc->Desc_1_0.Flags = (RootSignatureFlags)Value;
    break;
  case RootSignatureVersion::Version_1_1:
  case RootSignatureVersion::Version_1_2:
    llvm_unreachable("Not implemented yet");
    break;
  }
  return false;
}

bool root_signature::MetadataParser::ParseRootSignatureElement(
    MDNode *Element, VersionedRootSignatureDesc *Desc) {
  MDString *ElementText = cast<MDString>(Element->getOperand(0));

  assert(ElementText != nullptr && "First preoperty of element is not ");

  RootSignatureElementKind ElementKind =
      StringSwitch<RootSignatureElementKind>(ElementText->getString())
          .Case("RootFlags", RootSignatureElementKind::RootFlags)
          .Case("RootConstants", RootSignatureElementKind::RootConstants)
          .Case("RootCBV", RootSignatureElementKind::RootDescriptor)
          .Case("RootSRV", RootSignatureElementKind::RootDescriptor)
          .Case("RootUAV", RootSignatureElementKind::RootDescriptor)
          .Case("Sampler", RootSignatureElementKind::RootDescriptor)
          .Case("DescriptorTable", RootSignatureElementKind::DescriptorTable)
          .Case("StaticSampler", RootSignatureElementKind::StaticSampler)
          .Default(RootSignatureElementKind::None);

  switch (ElementKind) {

  case RootSignatureElementKind::RootFlags: {
    return ParseRootFlags(Element, Desc);
    break;
  }

  case RootSignatureElementKind::RootConstants:
  case RootSignatureElementKind::RootDescriptor:
  case RootSignatureElementKind::DescriptorTable:
  case RootSignatureElementKind::StaticSampler:
  case RootSignatureElementKind::None:
    llvm_unreachable("Not Implemented yet");
    break;
  }

  return true;
}
} // namespace dxil
} // namespace llvm
