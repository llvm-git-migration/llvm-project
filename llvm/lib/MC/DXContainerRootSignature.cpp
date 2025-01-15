//===- DXContainerRootSignature.cpp - DXIL Root Signature helper objects
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

#include "llvm/MC/DXContainerRootSignature.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SwapByteOrder.h"
#include <cassert>
#include <cstdint>

using namespace llvm;
using namespace llvm::mcdxbc;

void RootSignatureDescWriter::write(raw_ostream &OS, uint32_t Version) {
  dxbc::RootSignatureDesc Out{Desc->Version, Desc->Flags};

  if (sys::IsBigEndianHost) {
    Out.swapBytes();
  }

  OS.write(reinterpret_cast<const char *>(&Out), sizeof(RootSignatureDesc));
}
