//===- CoroSplit.cpp - Converts a coroutine into a state machine ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Coroutines/CoroAnnotationElide.h"

#include "llvm/Analysis/LazyCallGraph.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"

#include <cassert>

using namespace llvm;

#define DEBUG_TYPE "coro-annotation-elide"

#define CORO_MUST_ELIDE_ANNOTATION "coro_must_elide"

static bool processFunction(Function &F) {
  for (auto &I : instructions(F)) {
    if (auto *CB = dyn_cast<CallBase>(&I)) {
      if (CB->hasAnnotationMetadata(CORO_MUST_ELIDE_ANNOTATION)) {
        llvm::dbgs() << "hasAnnotation: ";
        CB->dump();
      }
    }
  }
  return false;
}

PreservedAnalyses CoroAnnotationElidePass::run(LazyCallGraph::SCC &C,
                                               CGSCCAnalysisManager &AM,
                                               LazyCallGraph &CG,
                                               CGSCCUpdateResult &UR) {
  // NB: One invariant of a valid LazyCallGraph::SCC is that it must contain a
  //     non-zero number of nodes, so we assume that here and grab the first
  //     node's function's module.
  Module &M = *C.begin()->getFunction().getParent();

  // Find coroutines for processing.
  SmallVector<LazyCallGraph::Node *> Coroutines;
  for (LazyCallGraph::Node &N : C) {
    Function &F = N.getFunction();
    processFunction(F);
  }
  return PreservedAnalyses::none();
}
