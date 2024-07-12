//===- CoroSplit.cpp - Converts a coroutine into a state machine ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Coroutines/CoroAnnotationElide.h"

#include "llvm/IR/Analysis.h"
#include "llvm/Analysis/LazyCallGraph.h"

#include <cassert>

llvm::PreservedAnalyses
llvm::CoroAnnotationElidePass::run(LazyCallGraph::SCC &C,
                                     CGSCCAnalysisManager &AM,
                                     LazyCallGraph &CG, CGSCCUpdateResult &UR) {
  // // NB: One invariant of a valid LazyCallGraph::SCC is that it must contain a
  // //     non-zero number of nodes, so we assume that here and grab the first
  // //     node's function's module.
  // Module &M = *C.begin()->getFunction().getParent();
  // auto &FAM =
  //     AM.getResult<FunctionAnalysisManagerCGSCCProxy>(C, CG).getManager();

  // // Find coroutines for processing.
  // SmallVector<LazyCallGraph::Node *> Coroutines;
  // for (LazyCallGraph::Node &N : C)
  //   if (N.getFunction().isPresplitCoroutine())
  //     Coroutines.push_back(&N);
  // // Split all the coroutines.
  // for (LazyCallGraph::Node *N : Coroutines) {
  //   Function &F = N->getFunction();

  // }

  return PreservedAnalyses::none();
}
