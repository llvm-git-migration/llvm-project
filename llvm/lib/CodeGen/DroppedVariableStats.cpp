///===- DroppedVariableStats.cpp - Opt Diagnostic -*- C++ -*---------------===//
///
/// Part of the LLVM Project, under the Apache License v2.0 with LLVM
/// Exceptions. See https://llvm.org/LICENSE.txt for license information.
/// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
///
///===---------------------------------------------------------------------===//
/// \file
/// Dropped Variable Statistics for Debug Information. Reports any number
/// of #dbg_value that get dropped due to an optimization pass.
///
///===---------------------------------------------------------------------===//

#include "llvm/CodeGen/DroppedVariableStats.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"

using namespace llvm;

bool DroppedVariableStats::isScopeChildOfOrEqualTo(DIScope *Scope,
                                                   const DIScope *DbgValScope) {
  while (Scope != nullptr) {
    if (VisitedScope.find(Scope) == VisitedScope.end()) {
      VisitedScope.insert(Scope);
      if (Scope == DbgValScope) {
        VisitedScope.clear();
        return true;
      }
      Scope = Scope->getScope();
    } else {
      VisitedScope.clear();
      return false;
    }
  }
  return false;
}

bool DroppedVariableStats::isInlinedAtChildOfOrEqualTo(
    const DILocation *InlinedAt, const DILocation *DbgValInlinedAt) {
  if (DbgValInlinedAt == InlinedAt)
    return true;
  if (!DbgValInlinedAt)
    return false;
  if (!InlinedAt)
    return false;
  auto *IA = InlinedAt;
  while (IA) {
    if (IA == DbgValInlinedAt)
      return true;
    IA = IA->getInlinedAt();
  }
  return false;
}

void DroppedVariableStatsIR::runOnFunction(const Function *F, bool Before) {
  auto &DebugVariables = DebugVariablesStack.back()[F];
  auto &VarIDSet = (Before ? DebugVariables.DebugVariablesBefore
                           : DebugVariables.DebugVariablesAfter);
  auto &InlinedAtsMap = InlinedAts.back();
  auto FuncName = F->getName();
  if (Before)
    InlinedAtsMap.try_emplace(FuncName, DenseMap<VarID, DILocation *>());
  VarIDSet = DenseSet<VarID>();
  for (const auto &I : instructions(F)) {
    for (DbgRecord &DR : I.getDbgRecordRange()) {
      if (auto *Dbg = dyn_cast<DbgVariableRecord>(&DR)) {
        auto *DbgVar = Dbg->getVariable();
        auto DbgLoc = DR.getDebugLoc();
        VarID Key{DbgVar->getScope(), DbgLoc->getInlinedAtScope(), DbgVar};
        VarIDSet.insert(Key);
        if (Before)
          InlinedAtsMap[FuncName].try_emplace(Key, DbgLoc.getInlinedAt());
      }
    }
  }
}

void DroppedVariableStatsIR::calculateDroppedVarStatsOnFunction(
    const Function *F, StringRef PassID, std::string FuncOrModName,
    std::string PassLevel) {
  unsigned DroppedCount = 0;
  StringRef FuncName = F->getName();
  DebugVariables &DbgVariables = DebugVariablesStack.back()[F];
  DenseSet<VarID> &DebugVariablesBeforeSet = DbgVariables.DebugVariablesBefore;
  DenseSet<VarID> &DebugVariablesAfterSet = DbgVariables.DebugVariablesAfter;
  DenseMap<VarID, DILocation *> &InlinedAtsMap = InlinedAts.back()[FuncName];
  // Find an Instruction that shares the same scope as the dropped #dbg_value or
  // has a scope that is the child of the scope of the #dbg_value, and has an
  // inlinedAt equal to the inlinedAt of the #dbg_value or it's inlinedAt chain
  // contains the inlinedAt of the #dbg_value, if such an Instruction is found,
  // debug information is dropped.
  for (VarID Var : DebugVariablesBeforeSet) {
    if (DebugVariablesAfterSet.contains(Var))
      continue;
    const DIScope *DbgValScope = std::get<0>(Var);
    for (const auto &I : instructions(F)) {
      auto *DbgLoc = I.getDebugLoc().get();
      if (!DbgLoc)
        continue;

      auto *Scope = DbgLoc->getScope();
      if (isScopeChildOfOrEqualTo(Scope, DbgValScope)) {
        if (isInlinedAtChildOfOrEqualTo(DbgLoc->getInlinedAt(),
                                        InlinedAtsMap[Var])) {
          // Found another instruction in the variable's scope, so there exists
          // a break point at which the variable could be observed. Count it as
          // dropped.
          DroppedCount++;
          break;
        }
      }
    }
    removeVarFromAllSets(Var, F);
  }
  if (DroppedCount > 0) {
    llvm::outs() << PassLevel << ", " << PassID << ", " << DroppedCount << ", "
                 << FuncOrModName << "\n";
    PassDroppedVariables = true;
  } else
    PassDroppedVariables = false;
}

void DroppedVariableStatsIR::runOnModule(const Module *M, bool Before) {
  for (auto &F : *M)
    runOnFunction(&F, Before);
}

void DroppedVariableStatsIR::calculateDroppedVarStatsOnModule(
    const Module *M, StringRef PassID, std::string FuncOrModName,
    std::string PassLevel) {
  for (auto &F : *M) {
    calculateDroppedVarStatsOnFunction(&F, PassID, FuncOrModName, PassLevel);
  }
}

void DroppedVariableStatsIR::registerCallbacks(
    PassInstrumentationCallbacks &PIC) {
  if (!DroppedVariableStatsEnabled)
    return;

  PIC.registerBeforeNonSkippedPassCallback(
      [this](StringRef P, Any IR) { return runBeforePass(IR); });
  PIC.registerAfterPassCallback(
      [this](StringRef P, Any IR, const PreservedAnalyses &PA) {
        return runAfterPass(P, IR);
      });
  PIC.registerAfterPassInvalidatedCallback(
      [this](StringRef P, const PreservedAnalyses &PA) { return cleanup(); });
}
