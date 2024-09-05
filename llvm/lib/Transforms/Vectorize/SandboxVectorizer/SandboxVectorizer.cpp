//===- SandboxVectorizer.cpp - Vectorizer based on Sandbox IR -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/SandboxVectorizer.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

#define SV_NAME "sandbox-vectorizer"
#define DEBUG_TYPE "SBVec"

cl::opt<bool>
    SBVecDisable("sbvec-disable", cl::init(false), cl::Hidden,
                 cl::desc("Disable the Sandbox Vectorization passes"));

PreservedAnalyses SandboxVectorizerPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  TTI = &AM.getResult<TargetIRAnalysis>(F);

  bool Changed = runImpl(F);
  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}

bool SandboxVectorizerPass::runImpl(Function &F) {
  if (SBVecDisable)
    return false;

  // If the target claims to have no vector registers don't attempt
  // vectorization.
  if (!TTI->getNumberOfRegisters(TTI->getRegisterClassForType(true))) {
    LLVM_DEBUG(dbgs() << "SBVec: Target has no vector registers, abort.\n");
    return false;
  }

  // Don't vectorize when the attribute NoImplicitFloat is used.
  if (F.hasFnAttribute(Attribute::NoImplicitFloat))
    return false;

  sandboxir::Context Ctx(F.getContext());

  LLVM_DEBUG(dbgs() << "SBVec: Analyzing blocks in " << F.getName() << ".\n");

  // Create SandboxIR for `F`.
  sandboxir::Function &SBF = *Ctx.createFunction(&F);

  // TODO: Initialize SBVec Pass Manager
  (void)SBF;

  return false;
}
