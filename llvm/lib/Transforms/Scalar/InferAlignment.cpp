//===- InferAlignment.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Infer alignment for load, stores and other memory operations based on
// trailing zero known bits information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/InferAlignment.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Instructions.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

DenseMap<Value *, Value *> ValueToBasePtr;

static bool tryToImproveAlign(
    const DataLayout &DL, Instruction *I,
    function_ref<Align(Value *PtrOp, Align OldAlign, Align PrefAlign)> Fn) {

  if (auto *PtrOp = getLoadStorePointerOperand(I)) {
    Align OldAlign = getLoadStoreAlignment(I);
    Align PrefAlign = DL.getPrefTypeAlign(getLoadStoreType(I));

    Align NewAlign = Fn(PtrOp, OldAlign, PrefAlign);
    if (NewAlign > OldAlign) {
      setLoadStoreAlignment(I, NewAlign);
      return true;
    }
  }

  // TODO: Also handle memory intrinsics.
  return false;
}

static bool needEnforceAlignment(Value *PtrOp, Instruction *I, Align PrefAlign,
                                 const DataLayout &DL) {
  auto it = ValueToBasePtr.find(PtrOp);
  if (it != ValueToBasePtr.end()) {
    Value *V = it->second;
    Align CurrentAlign;
    if (auto Alloca = dyn_cast<AllocaInst>(V))
      CurrentAlign = Alloca->getAlign();
    if (auto GO = dyn_cast<GlobalObject>(V))
      CurrentAlign = GO->getPointerAlignment(DL);

    if (PrefAlign <= CurrentAlign) {
      setLoadStoreAlignment(I, CurrentAlign);
      return false;
    }
  }

  return true;
}

bool inferAlignment(Function &F, AssumptionCache &AC, DominatorTree &DT) {
  const DataLayout &DL = F.getDataLayout();
  bool Changed = false;

  // Enforce preferred type alignment if possible. We do this as a separate
  // pass first, because it may improve the alignments we infer below.
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      Changed |= tryToImproveAlign(
          DL, &I, [&](Value *PtrOp, Align OldAlign, Align PrefAlign) {
            if (needEnforceAlignment(PtrOp, &I, PrefAlign, DL) &&
                PrefAlign > OldAlign) {
              Align NewAlign = tryEnforceAlignment(PtrOp, PrefAlign, DL);
              if (NewAlign > OldAlign) {
                ValueToBasePtr[PtrOp] = PtrOp->stripPointerCasts();
                return NewAlign;
              }
            }

            return OldAlign;
          });
    }
  }

  // Compute alignment from known bits.
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      Changed |= tryToImproveAlign(
          DL, &I, [&](Value *PtrOp, Align OldAlign, Align PrefAlign) {
            return getOrEnforceKnownAlignment(PtrOp, MaybeAlign(), DL, &I, &AC,
                                              &DT);
          });
    }
  }

  return Changed;
}

PreservedAnalyses InferAlignmentPass::run(Function &F,
                                          FunctionAnalysisManager &AM) {
  AssumptionCache &AC = AM.getResult<AssumptionAnalysis>(F);
  DominatorTree &DT = AM.getResult<DominatorTreeAnalysis>(F);
  inferAlignment(F, AC, DT);
  // Changes to alignment shouldn't invalidated analyses.
  return PreservedAnalyses::all();
}
