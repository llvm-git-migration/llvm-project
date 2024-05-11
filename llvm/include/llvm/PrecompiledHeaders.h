//===-- llvm/PrecompiledHeaders.h - Precompiled Headers ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the most common headers used within the LLVM library. 
/// It is intended to be used as a precompiled header.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_PRECOMPILEDHEADERS_H
#define LLVM_PRECOMPILEDHEADERS_H

#include "ADT/ArrayRef.h"

#include "CodeGen/SelectionDAG.h"
#include "CodeGen/TargetInstrInfo.h"

#include "IR/IntrinsicInst.h"
#include "IR/PassManager.h"

#include "MC/MCContext.h"

#endif // LLVM_PRECOMPILEDHEADERS_H