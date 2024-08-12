//===- lib/Target/AMDGPU/AMDGPUCodeGenPassBuilder.h -----------*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUCODEGENPASSBUILDER_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUCODEGENPASSBUILDER_H

#include "llvm/MC/MCStreamer.h"
#include "llvm/Passes/CodeGenPassBuilder.h"

namespace llvm {

class GCNTargetMachine;

class AMDGPUCodeGenPassBuilder
    : public CodeGenPassBuilder<AMDGPUCodeGenPassBuilder, GCNTargetMachine> {
public:
  using Base = CodeGenPassBuilder<AMDGPUCodeGenPassBuilder, GCNTargetMachine>;

  AMDGPUCodeGenPassBuilder(GCNTargetMachine &TM,
                           const CGPassBuilderOption &Opts,
                           PassInstrumentationCallbacks *PIC);
  void addCodeGenPrepare(AddIRPass &) const;
  void addPreISel(AddIRPass &addPass) const;
  void addAsmPrinter(AddMachinePass &, CreateMCStreamer) const;
  Error addInstSelector(AddMachinePass &) const;

  /// Check if a pass is enabled given \p Opt option. The option always
  /// overrides defaults if explicitly used. Otherwise its default will
  /// be used given that a pass shall work at an optimization \p Level
  /// minimum.
  bool isPassEnabled(const cl::opt<bool> &Opt,
                     CodeGenOptLevel Level = CodeGenOptLevel::Default) const;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUCODEGENPASSBUILDER_H
