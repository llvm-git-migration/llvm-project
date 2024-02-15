//===-- AMDGPUCloneModuleLDSPass.cpp ------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The purpose of this pass is to ensure that the combined module contains
// as many LDS global variables as there are kernels that (indirectly) access
// them. As LDS variables behave like C++ static variables, it is important that
// each partition contains a unique copy of the variable on a per kernel basis.
// This representation also prepares the combined module to eliminate
// cross-module dependencies of LDS variables.
//
// This pass operates as follows:
// 1. Firstly, traverse the call graph from each kernel to determine the number
//    of kernels calling each device function.
// 2. For each LDS global variable GV, determine the function F that defines it.
//    Collect it's caller functions. Clone F and GV, and finally insert a
//    call/invoke instruction in each caller function.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-clone-module-lds"

static cl::opt<unsigned int> MaxCountForClonedFunctions(
    "clone-lds-functions-max-count", cl::init(16), cl::Hidden,
    cl::desc("Specify a limit to the number of clones of a function"));

/// Return the function that defines \p GV
/// \param GV The global variable in question
/// \return The function defining \p GV
static Function *getFunctionDefiningGV(GlobalVariable &GV) {
  SmallVector<User *> Worklist(GV.users());
  while (!Worklist.empty()) {
    User *U = Worklist.pop_back_val();
    if (auto *Inst = dyn_cast<Instruction>(U))
      return Inst->getFunction();
    if (auto *Op = dyn_cast<Operator>(U))
      append_range(Worklist, Op->users());
  }
  llvm_unreachable("GV must be used in a function.");
};

/// Replace all references to \p OldGV in \p NewF with \p NewGV
/// \param OldGV The global variable to be replaced
/// \param NewGV The global variable taking the place of \p OldGV
/// \param NewF The function in which the replacement occurs
static void replaceUsesOfWith(GlobalVariable *OldGV, GlobalVariable *NewGV,
                              Function *NewF) {
  // ReplaceOperatorUses takes in an instruction Inst, which is assumed to
  // contain OldGV as an operator, inserts an instruction correponding the
  // OldGV-operand and update Inst accordingly.
  auto ReplaceOperatorUses = [&OldGV, &NewGV](Instruction *Inst) {
    Inst->replaceUsesOfWith(OldGV, NewGV);
    SmallVector<Value *, 8> Worklist(Inst->operands());
    while (!Worklist.empty()) {
      auto *V = Worklist.pop_back_val();
      if (auto *I = dyn_cast<AddrSpaceCastOperator>(V)) {
        auto *Cast = new AddrSpaceCastInst(NewGV, I->getType(), "", Inst);
        Inst->replaceUsesOfWith(I, Cast);
      } else if (auto *I = dyn_cast<GEPOperator>(V)) {
        SmallVector<Value *, 8> Indices(I->indices());
        auto *GEP = GetElementPtrInst::Create(NewGV->getValueType(), NewGV,
                                              Indices, "", Inst);
        Inst->replaceUsesOfWith(I, GEP);
      }
    }
  };

  // Collect all user instructions of OldGV using a Worklist algorithm.
  // If a user is an operator, collect the instruction wrapping
  // the operator.
  SmallVector<Instruction *, 8> InstsToReplace;
  SmallVector<User *, 8> UsersWorklist(OldGV->users());
  while (!UsersWorklist.empty()) {
    auto *U = UsersWorklist.pop_back_val();
    if (auto *Inst = dyn_cast<Instruction>(U)) {
      InstsToReplace.push_back(Inst);
    } else if (auto *Op = dyn_cast<Operator>(U)) {
      append_range(UsersWorklist, Op->users());
    }
  }

  // Replace all occurences of OldGV in NewF
  DenseSet<Instruction *> ReplacedInsts;
  while (!InstsToReplace.empty()) {
    auto *Inst = InstsToReplace.pop_back_val();
    if (Inst->getFunction() != NewF || ReplacedInsts.contains(Inst))
      continue;
    ReplaceOperatorUses(Inst);
    ReplacedInsts.insert(Inst);
  }
};

PreservedAnalyses AMDGPUCloneModuleLDSPass::run(Module &M,
                                                ModuleAnalysisManager &AM) {
  if (MaxCountForClonedFunctions.getValue() == 1)
    return PreservedAnalyses::all();

  bool Changed = false;
  auto &CG = AM.getResult<CallGraphAnalysis>(M);

  // For each function in the call graph, determine the number
  // of ancestor-caller kernels.
  DenseMap<Function *, unsigned int> KernelRefsToFuncs;
  for (auto &Fn : M) {
    if (Fn.getCallingConv() != CallingConv::AMDGPU_KERNEL)
      continue;
    for (auto I = df_begin(&CG), E = df_end(&CG); I != E; ++I)
      if (auto *F = I->getFunction())
        KernelRefsToFuncs[F]++;
  }

  DenseMap<GlobalVariable *, Function *> GVToFnMap;
  LLVMContext &Ctx = M.getContext();
  IRBuilder<> IRB(Ctx);
  for (auto &GV : M.globals()) {
    if (GVToFnMap.contains(&GV) ||
        GV.getType()->getPointerAddressSpace() != AMDGPUAS::LOCAL_ADDRESS ||
        !GV.hasInitializer())
      continue;

    auto *OldF = getFunctionDefiningGV(GV);
    GVToFnMap.insert({&GV, OldF});
    LLVM_DEBUG(dbgs() << "Found LDS " << GV.getName() << " used in function "
                      << OldF->getName() << '\n');

    // Collect all caller functions of OldF. Each of them must call it's
    // corresponding clone of OldF.
    SmallVector<std::pair<Instruction *, SmallVector<Value *>>>
        InstsCallingOldF;
    for (auto &I : OldF->uses()) {
      User *U = I.getUser();
      SmallVector<Value *> Args;
      if (auto *CI = dyn_cast<CallInst>(U)) {
        append_range(Args, CI->args());
        InstsCallingOldF.push_back({CI, Args});
      } else if (auto *II = dyn_cast<InvokeInst>(U)) {
        append_range(Args, II->args());
        InstsCallingOldF.push_back({II, Args});
      }
    }

    // Create as many clones of the function containing LDS global as
    // there are kernels calling the function (including the function
    // already defining the LDS global). Respectively, clone the
    // LDS global and the call instructions to the function.
    LLVM_DEBUG(dbgs() << "\tFunction is referenced by "
                      << KernelRefsToFuncs[OldF] << " kernels.\n");
    for (unsigned int ID = 0;
         ID + 1 < std::min(KernelRefsToFuncs[OldF],
                           MaxCountForClonedFunctions.getValue());
         ++ID) {
      // Clone function
      ValueToValueMapTy VMap;
      auto *NewF = CloneFunction(OldF, VMap);
      NewF->setName(OldF->getName() + ".clone." + to_string(ID));
      LLVM_DEBUG(dbgs() << "Inserting function clone with name "
                        << NewF->getName() << '\n');

      // Clone LDS global variable
      auto *NewGV = new GlobalVariable(
          M, GV.getValueType(), GV.isConstant(), GlobalValue::InternalLinkage,
          UndefValue::get(GV.getValueType()),
          GV.getName() + ".clone." + to_string(ID), &GV,
          GlobalValue::NotThreadLocal, AMDGPUAS::LOCAL_ADDRESS, false);
      NewGV->copyAttributesFrom(&GV);
      NewGV->copyMetadata(&GV, 0);
      NewGV->setComdat(GV.getComdat());
      replaceUsesOfWith(&GV, NewGV, NewF);
      LLVM_DEBUG(dbgs() << "Inserting LDS clone with name " << NewGV->getName()
                        << "\n");

      // Create a new CallInst to call the cloned function
      for (auto [Inst, Args] : InstsCallingOldF) {
        IRB.SetInsertPoint(Inst);
        Instruction *I;
        if (isa<CallInst>(Inst))
          I = IRB.CreateCall(NewF, Args,
                             Inst->getName() + ".clone." + to_string(ID));
        else if (auto *II = dyn_cast<InvokeInst>(Inst))
          I = IRB.CreateInvoke(NewF, II->getNormalDest(), II->getUnwindDest(),
                               Args, II->getName() + ".clone" + to_string(ID));
        LLVM_DEBUG(dbgs() << "Inserting inst: " << *I << '\n');
      }
      Changed = true;
    }
  }
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
