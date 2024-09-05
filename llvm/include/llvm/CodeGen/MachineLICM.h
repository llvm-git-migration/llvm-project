
//===- llvm/CodeGen/MachineLICM.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINELICM_H
#define LLVM_CODEGEN_MACHINELICM_H

#include "llvm/CodeGen/MachinePassManager.h"

namespace llvm {

template <typename DerivedT, bool PreRegAlloc>
class MachineLICMBasePass : public PassInfoMixin<DerivedT> {
public:
  PreservedAnalyses run(MachineFunction &MF,
                        MachineFunctionAnalysisManager &MFAM);
};
class EarlyMachineLICMPass
    : public MachineLICMBasePass<EarlyMachineLICMPass, true> {};

class MachineLICMPass : public MachineLICMBasePass<MachineLICMPass, false> {};

template class MachineLICMBasePass<EarlyMachineLICMPass, true>;
template class MachineLICMBasePass<MachineLICMPass, false>;

} // namespace llvm

#endif // LLVM_CODEGEN_MACHINELICM_H
