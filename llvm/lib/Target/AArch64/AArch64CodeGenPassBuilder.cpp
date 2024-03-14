//===-- AArch64CodeGenPassBuilder.cpp -----------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains AArch64 CodeGen pipeline builder.
/// TODO: Port CodeGen passes to new pass manager.
//===----------------------------------------------------------------------===//

#include "AArch64LoopIdiomTransform.h"
#include "AArch64TargetMachine.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Passes/PassBuilder.h"

using namespace llvm;

void AArch64TargetMachine::registerPassBuilderCallbacks(
    PassBuilder &PB, bool PopulateClassToPassNames) {
  if (PopulateClassToPassNames) {
    auto *PIC = PB.getPassInstrumentationCallbacks();
#define LOOP_PASS(NAME, CREATE_PASS)                                           \
  PIC->addClassToPassName(decltype(CREATE_PASS)::name(), NAME);
#include "AArch64PassRegistry.def"
  }

  PB.registerPipelineParsingCallback(
      [](StringRef Name, LoopPassManager &LPM,
         ArrayRef<PassBuilder::PipelineElement>) {
#define LOOP_PASS(NAME, CREATE_PASS)                                           \
  if (Name == NAME) {                                                          \
    LPM.addPass(CREATE_PASS);                                                  \
    return true;                                                               \
  }
#include "AArch64PassRegistry.def"
        return false;
      });

  PB.registerLateLoopOptimizationsEPCallback(
      [=](LoopPassManager &LPM, OptimizationLevel Level) {
        LPM.addPass(AArch64LoopIdiomTransformPass());
      });
}
