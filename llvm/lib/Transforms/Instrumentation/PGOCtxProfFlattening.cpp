//===- PGOCtxProfFlattening.cpp - Contextual Instr. Flattening ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Flattens the contextual profile and lowers it to MD_prof.
// This should happen after all IPO (which is assumed to have maintained the
// contextual profile) happened. Flattening consists of summing the values at
// the same index of the counters belonging to all the contexts of a function.
// The lowering consists of materializing the counter values to function
// entrypoint counts and branch probabilities.
//
// This pass also removes contextual instrumentation, which has been kept around
// to facilitate its functionality.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/PGOCtxProfFlattening.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/CtxProfAnalysis.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/ProfileSummary.h"
#include "llvm/ProfileData/ProfileCommon.h"
#include "llvm/Transforms/Instrumentation/PGOInstrumentation.h"
#include "llvm/Transforms/Scalar/DCE.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

namespace {

class Solver final {
  struct BBInfo;
  struct EdgeInfo {
    BBInfo *const Src;
    BBInfo *const Dest;
    std::optional<uint64_t> Count;

    explicit EdgeInfo(BBInfo *Src, BBInfo *Dest) : Src(Src), Dest(Dest) {}
  };

  struct BBInfo {
    std::optional<uint64_t> Count;
    SmallVector<EdgeInfo *> OutEdges;
    SmallVector<EdgeInfo *> InEdges;
    size_t UnknownCountOutEdges = 0;
    size_t UnknownCountInEdges = 0;

    uint64_t getEdgeSum(const SmallVector<EdgeInfo *> &Edges,
                        bool AssumeAllKnown) const {
      uint64_t Sum = 0;
      for (const auto *E : Edges)
        if (E)
          Sum += AssumeAllKnown ? *E->Count : E->Count.value_or(0U);
      return Sum;
    }

    void takeCountFrom(const SmallVector<EdgeInfo *> &Edges) {
      assert(!Count.has_value());
      Count = getEdgeSum(Edges, true);
    }
  };

  Function &F;
  std::map<const BasicBlock *, BBInfo> BBInfos;
  std::vector<EdgeInfo> EdgeInfos;
  InstrProfSummaryBuilder &PB;

  void setSingleUnknownEdgeCount(SmallVector<EdgeInfo *> &Edges,
                                 uint64_t Value) {
    EdgeInfo *E = nullptr;
    for (auto *I : Edges)
      if (I && !I->Count.has_value()) {
        E = I;
#ifdef NDEBUG
        break;
#else
        assert((!E || E == I) &&
               "Expected exactly one edge to have an unknown count, "
               "found a second one");
        continue;
#endif
      }
    assert(E && "Expected exactly one edge to have an unknown count");
    assert(!E->Count.has_value());
    E->Count = Value;
    assert(E->Src->UnknownCountOutEdges > 0);
    assert(E->Dest->UnknownCountInEdges > 0);
    --E->Src->UnknownCountOutEdges;
    --E->Dest->UnknownCountInEdges;
  }

  void solve(const SmallVectorImpl<uint64_t> &Counters) {
    for (const auto &BB : F) {
      if (auto *Ins = CtxProfAnalysis::getBBInstrumentation(
              const_cast<BasicBlock &>(BB)))
        BBInfos.find(&BB)->second.Count =
            Counters[Ins->getIndex()->getZExtValue()];
    }
    bool KeepGoing = true;
    while (KeepGoing) {
      KeepGoing = false;
      for (const auto &BB : reverse(F)) {
        auto &Info = BBInfos.find(&BB)->second;
        if (!Info.Count) {
          if (!succ_empty(&BB) && !Info.UnknownCountOutEdges) {
            Info.takeCountFrom(Info.OutEdges);
            KeepGoing = true;
          } else if (!BB.isEntryBlock() && !Info.UnknownCountInEdges) {
            Info.takeCountFrom(Info.InEdges);
            KeepGoing = true;
          }
        }
        if (Info.Count.has_value()) {
          if (Info.UnknownCountOutEdges == 1) {
            uint64_t KnownSum = Info.getEdgeSum(Info.OutEdges, false);
            uint64_t EdgeVal =
                *Info.Count > KnownSum ? *Info.Count - KnownSum : 0U;
            setSingleUnknownEdgeCount(Info.OutEdges, EdgeVal);
            KeepGoing = true;
          }
          if (Info.UnknownCountInEdges == 1) {
            uint64_t KnownSum = Info.getEdgeSum(Info.InEdges, false);
            uint64_t EdgeVal =
                *Info.Count > KnownSum ? *Info.Count - KnownSum : 0U;
            setSingleUnknownEdgeCount(Info.InEdges, EdgeVal);
            KeepGoing = true;
          }
        }
      }
    }
  }
  // The only criteria for exclusion is faux suspend -> exit edges in presplit
  // coroutines. The API serves for readability, currently.
  bool shouldExcludeEdge(const BasicBlock &Src, const BasicBlock &Dest) const {
    return llvm::isPresplitCoroSuspendExitEdge(Src, Dest);
  }

public:
  Solver(Function &F, InstrProfSummaryBuilder &PB) : F(F), PB(PB) {
    assert(!F.isDeclaration());
    size_t NrEdges = 0;
    for (const auto &BB : F) {
      auto [It, Ins] = BBInfos.insert({&BB, {}});
      (void)Ins;
      assert(Ins);
      NrEdges += llvm::count_if(successors(&BB), [&](const auto *Succ) {
        return !shouldExcludeEdge(BB, *Succ);
      });
      It->second.InEdges.reserve(pred_size(&BB));
      It->second.OutEdges.resize(succ_size(&BB));
    }
    EdgeInfos.reserve(NrEdges);
    for (const auto &BB : F) {
      auto &Info = BBInfos.find(&BB)->second;
      for (auto I = 0U; I < BB.getTerminator()->getNumSuccessors(); ++I) {
        const auto *Succ = BB.getTerminator()->getSuccessor(I);
        if (!shouldExcludeEdge(BB, *Succ)) {
          auto &EI = EdgeInfos.emplace_back(&BBInfos.find(&BB)->second,
                                            &BBInfos.find(Succ)->second);
          Info.OutEdges[I] = &EI;
          ++Info.UnknownCountOutEdges;
          BBInfos.find(Succ)->second.InEdges.push_back(&EI);
          ++BBInfos.find(Succ)->second.UnknownCountInEdges;
        }
      }
    }
  }

  void assignProfData(const SmallVectorImpl<uint64_t> &Counters) {
    assert(!Counters.empty());
    solve(Counters);
    F.setEntryCount(Counters[0]);
    PB.addEntryCount(Counters[0]);

    for (auto &BB : F) {
      if (succ_size(&BB) < 2)
        continue;
      auto *Term = BB.getTerminator();
      SmallVector<uint64_t, 2> EdgeCounts(Term->getNumSuccessors(), 0);
      uint64_t MaxCount = 0;
      const auto &BBInfo = BBInfos.find(&BB)->second;
      for (unsigned SuccIdx = 0, Size = BBInfo.OutEdges.size(); SuccIdx < Size;
           ++SuccIdx) {
        const auto *E = BBInfo.OutEdges[SuccIdx];
        if (!E)
          continue;
        uint64_t EdgeCount = *E->Count;
        if (EdgeCount > MaxCount)
          MaxCount = EdgeCount;
        EdgeCounts[SuccIdx] = EdgeCount;
        PB.addInternalCount(EdgeCount);
      }

      if (MaxCount == 0)
        F.getContext().emitError(
            "[ctx-prof] Encountered a BB with more than one successor, where "
            "all outgoing edges have a 0 count. This occurs in non-exiting "
            "functions (message pumps, usually) which are not supported in the "
            "contextual profiling case");
      setProfMetadata(F.getParent(), Term, EdgeCounts, MaxCount);
    }
  }
};

bool areAllBBsReachable(const Function &F, FunctionAnalysisManager &FAM) {
  auto &DT = FAM.getResult<DominatorTreeAnalysis>(const_cast<Function &>(F));
  for (const auto &BB : F)
    if (!DT.isReachableFromEntry(&BB))
      return false;
  return true;
}

void clearColdFunctionProfile(Function &F) {
  for (auto &BB : F)
    BB.getTerminator()->setMetadata(LLVMContext::MD_prof, nullptr);
  F.setEntryCount(0U);
}

void removeInstrumentation(Function &F) {
  for (auto &BB : F)
    for (auto &I : llvm::make_early_inc_range(BB))
      if (isa<InstrProfCntrInstBase>(I))
        I.eraseFromParent();
}

} // namespace

PreservedAnalyses PGOCtxProfFlattening::run(Module &M,
                                            ModuleAnalysisManager &MAM) {
  auto &CtxProf = MAM.getResult<CtxProfAnalysis>(M);
  if (!CtxProf)
    return PreservedAnalyses::all();

  const auto FlattenedProfile = CtxProf.flatten();

  InstrProfSummaryBuilder PB(ProfileSummaryBuilder::DefaultCutoffs);
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;

    if (!areAllBBsReachable(F,
                            MAM.getResult<FunctionAnalysisManagerModuleProxy>(M)
                                .getManager())) {
      M.getContext().emitError(
          "[ctx-prof] Function has unreacheable basic blocks: " + F.getName());
      continue;
    }

    const auto &FlatProfile =
        FlattenedProfile.lookup(AssignGUIDPass::getGUID(F));
    if (FlatProfile.empty())
      clearColdFunctionProfile(F);
    else {
      Solver S(F, PB);
      S.assignProfData(FlatProfile);
    }
    removeInstrumentation(F);
  }

  auto &PSI = MAM.getResult<ProfileSummaryAnalysis>(M);

  PSI.overrideSummary(PB.getSummary());
  PSI.computeThresholds();

  return PreservedAnalyses::none();
}