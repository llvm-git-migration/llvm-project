//===- JumpTableToSwitch.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar/JumpTableToSwitch.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;
using namespace PatternMatch;

static cl::opt<unsigned>
    JumpTableSizeThreshold("jump-table-to-switch-size-threshold", cl::Hidden,
                           cl::desc("Only split jump tables with size less or "
                                    "equal than JumpTableSizeThreshold."),
                           cl::init(10));

#define DEBUG_TYPE "jump-table-to-switch"

namespace {
struct JumpTableTy {
  Value *Index;
  SmallVector<Function *, 5> Funcs;
};
} // anonymous namespace

static std::optional<JumpTableTy> parseJumpTable(GetElementPtrInst *GEP) {
  if (!GEP || !GEP->isInBounds())
    return std::nullopt;
  ArrayType *ArrayTy = dyn_cast<ArrayType>(GEP->getSourceElementType());
  if (!ArrayTy || ArrayTy->getArrayNumElements() > JumpTableSizeThreshold)
    return std::nullopt;
  const uint64_t N = ArrayTy->getArrayNumElements();
  Constant *Ptr = dyn_cast<Constant>(GEP->getPointerOperand());
  if (!Ptr || !Ptr->getNumOperands())
    return std::nullopt;

  GlobalVariable *GV = dyn_cast<GlobalVariable>(Ptr);
  if (!GV || !GV->isConstant())
    return std::nullopt;

  Function &F = *GEP->getParent()->getParent();

  const DataLayout &DL = F.getParent()->getDataLayout();
  const unsigned BitWidth =
      DL.getIndexSizeInBits(GEP->getPointerAddressSpace());
  MapVector<Value *, APInt> VariableOffsets;
  APInt ConstantOffset(BitWidth, 0);
  if (!GEP->collectOffset(DL, BitWidth, VariableOffsets, ConstantOffset))
    return std::nullopt;
  if (VariableOffsets.empty() || VariableOffsets.size() > 1)
    return std::nullopt;
  unsigned Offset = ConstantOffset.getZExtValue();
  // TODO: consider supporting more general patterns
  if (Offset != 0)
    return std::nullopt;

  JumpTableTy JumpTable;
  JumpTable.Index = VariableOffsets.front().first;
  JumpTable.Funcs.assign(N, nullptr);
  PointerType *PtrTy =
      PointerType::get(F.getContext(), DL.getProgramAddressSpace());
  const unsigned PtrSizeBits = DL.getPointerTypeSizeInBits(PtrTy);
  const unsigned PtrSizeBytes = DL.getPointerTypeSize(PtrTy);
  for (uint64_t Index = 0; Index < N; ++Index) {
    APInt Offset(PtrSizeBits, Index * PtrSizeBytes);
    Constant *C = ConstantFoldLoadFromConst(
        cast<Constant>(GV->getOperand(0)),
        PointerType::get(F.getContext(), DL.getProgramAddressSpace()), Offset,
        DL);
    auto *Func = dyn_cast_or_null<Function>(C);
    if (!Func || Func->isDeclaration())
      return std::nullopt;
    JumpTable.Funcs[Index] = Func;
  }
  return JumpTable;
}

static BasicBlock *split(CallBase *CB, const JumpTableTy &JT,
                         DomTreeUpdater *DTU) {
  const bool IsVoid = CB->getType() == Type::getVoidTy(CB->getContext());

  SmallVector<DominatorTree::UpdateType, 8> DTUpdates;
  BasicBlock *BB = CB->getParent();
  BasicBlock *Tail =
      SplitBlock(BB, CB, DTU, nullptr, nullptr, BB->getName() + Twine(".tail"));
  DTUpdates.push_back({DominatorTree::Delete, BB, Tail});
  BB->getTerminator()->eraseFromParent();

  Function &F = *BB->getParent();
  BasicBlock *BBUnreachable = BasicBlock::Create(
      F.getContext(), "default.switch.case.unreachable", &F, Tail);
  IRBuilder<> BuilderUnreachable(BBUnreachable);
  BuilderUnreachable.CreateUnreachable();

  IRBuilder<> Builder(BB);
  SwitchInst *Switch = Builder.CreateSwitch(JT.Index, BBUnreachable);
  DTUpdates.push_back({DominatorTree::Insert, BB, BBUnreachable});

  IRBuilder<> BuilderTail(CB);
  PHINode *PHI =
      IsVoid ? nullptr : BuilderTail.CreatePHI(CB->getType(), JT.Funcs.size());

  for (auto [Index, Func] : llvm::enumerate(JT.Funcs)) {
    BasicBlock *B = BasicBlock::Create(Func->getContext(),
                                       "call." + Twine(Index), &F, Tail);
    DTUpdates.push_back({DominatorTree::Insert, BB, B});
    DTUpdates.push_back({DominatorTree::Insert, B, Tail});

    CallBase *Call = cast<CallBase>(CB->clone());
    Call->setCalledFunction(Func);
    Call->insertInto(B, B->end());
    Switch->addCase(
        cast<ConstantInt>(ConstantInt::get(JT.Index->getType(), Index)), B);
    BranchInst::Create(Tail, B);
    if (PHI)
      PHI->addIncoming(Call, B);
  }
  if (DTU)
    DTU->applyUpdates(DTUpdates);
  if (PHI)
    CB->replaceAllUsesWith(PHI);
  CB->eraseFromParent();
  return Tail;
}

PreservedAnalyses JumpTableToSwitchPass::run(Function &F,
                                             FunctionAnalysisManager &AM) {
  DominatorTree *DT = AM.getCachedResult<DominatorTreeAnalysis>(F);
  PostDominatorTree *PDT = AM.getCachedResult<PostDominatorTreeAnalysis>(F);
  std::unique_ptr<DomTreeUpdater> DTU;
  bool Changed = false;
  for (BasicBlock &BB : make_early_inc_range(F)) {
    BasicBlock *CurrentBB = &BB;
    while (CurrentBB) {
      BasicBlock *SplittedOutTail = nullptr;
      for (Instruction &I : make_early_inc_range(*CurrentBB)) {
        CallBase *CB = dyn_cast<CallBase>(&I);
        if (!CB || isa<IntrinsicInst>(CB) || CB->getCalledFunction() ||
            isa<InvokeInst>(CB) || CB->isMustTailCall())
          continue;

        Value *V;
        if (!match(CB->getCalledOperand(), m_Load(m_Value(V))))
          continue;
        // Skip volatile loads.
        if (cast<LoadInst>(CB->getCalledOperand())->isVolatile())
          continue;
        auto *GEP = dyn_cast<GetElementPtrInst>(V);
        if (!GEP)
          continue;

        std::optional<JumpTableTy> JumpTable = parseJumpTable(GEP);
        if (!JumpTable)
          continue;
        if ((DT || PDT) && !DTU)
          DTU = std::make_unique<DomTreeUpdater>(
              DT, PDT, DomTreeUpdater::UpdateStrategy::Lazy);
        SplittedOutTail = split(CB, *JumpTable, DTU.get());
        Changed = true;
        break;
      }
      CurrentBB = SplittedOutTail ? SplittedOutTail : nullptr;
    }
  }

  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  if (DT)
    PA.preserve<DominatorTreeAnalysis>();
  if (PDT)
    PA.preserve<PostDominatorTreeAnalysis>();
  return PA;
}
