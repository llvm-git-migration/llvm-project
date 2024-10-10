//===- DXILFlattenArrays.cpp - Flattens DXIL Arrays-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

///
/// \file This file contains a pass to flatten arrays for the DirectX Backend.
//
//===----------------------------------------------------------------------===//

#include "DXILFlattenArrays.h"
#include "DirectX.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/Local.h"
#include <cassert>
#include <cstdint>

#define DEBUG_TYPE "dxil-flatten-arrays"

using namespace llvm;

class DXILFlattenArraysLegacy : public ModulePass {

public:
  bool runOnModule(Module &M) override;
  DXILFlattenArraysLegacy() : ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  static char ID; // Pass identification.
};

struct GEPChainInfo {
  DenseMap<GetElementPtrInst *, SmallVector<ConstantInt *>> GEPToIndicesMap;
  DenseMap<GetElementPtrInst *, SmallVector<uint64_t>> GEPToDimmsMap;
  DenseMap<GetElementPtrInst *, GetElementPtrInst *> GEPChildToNewParentMap;
  DenseMap<Value *, GetElementPtrInst *> OperandToBaseGEPMap;
};

class DXILFlattenArraysVisitor
    : public InstVisitor<DXILFlattenArraysVisitor, bool> {
public:
  DXILFlattenArraysVisitor() {}
  bool visit(Function &F);
  // InstVisitor methods.  They return true if the instruction was scalarized,
  // false if nothing changed.
  bool visitGetElementPtrInst(GetElementPtrInst &GEPI);
  bool visitAllocaInst(AllocaInst &AI);
  bool visitInstruction(Instruction &I) { return false; }
  bool visitSelectInst(SelectInst &SI) { return false; }
  bool visitICmpInst(ICmpInst &ICI) { return false; }
  bool visitFCmpInst(FCmpInst &FCI) { return false; }
  bool visitUnaryOperator(UnaryOperator &UO) { return false; }
  bool visitBinaryOperator(BinaryOperator &BO) { return false; }
  bool visitCastInst(CastInst &CI) { return false; }
  bool visitBitCastInst(BitCastInst &BCI) { return false; }
  bool visitInsertElementInst(InsertElementInst &IEI) { return false; }
  bool visitExtractElementInst(ExtractElementInst &EEI) { return false; }
  bool visitShuffleVectorInst(ShuffleVectorInst &SVI) { return false; }
  bool visitPHINode(PHINode &PHI) { return false; }
  bool visitLoadInst(LoadInst &LI) { return false; }
  bool visitStoreInst(StoreInst &SI) { return false; }
  bool visitCallInst(CallInst &ICI) { return false; }
  bool visitFreezeInst(FreezeInst &FI) { return false; }

private:
  SmallVector<WeakTrackingVH, 32> PotentiallyDeadInstrs;
  GEPChainInfo GEPChain;
  bool finish();
  bool isMultiDimensionalArray(Type *T);
  ConstantInt *flattenIndices(ArrayRef<ConstantInt *> Indices,
                              ArrayRef<uint64_t> Dims, IRBuilder<> &Builder);
  unsigned getTotalElements(Type *ArrayTy);
  Type *getBaseElementType(Type *ArrayTy);
  void recursivelyCollectGEPs(
      GetElementPtrInst &CurrGEP, GetElementPtrInst &NewGEP,
      GEPChainInfo &GEPChain,
      SmallVector<ConstantInt *> Indices = SmallVector<ConstantInt *>(),
      SmallVector<uint64_t> Dims = SmallVector<uint64_t>());
  bool visitGetElementPtrInstInGEPChain(GetElementPtrInst &GEP);
};

bool DXILFlattenArraysVisitor::finish() {
  RecursivelyDeleteTriviallyDeadInstructionsPermissive(PotentiallyDeadInstrs);
  return true;
}

bool DXILFlattenArraysVisitor::isMultiDimensionalArray(Type *T) {
  if (ArrayType *ArrType = dyn_cast<ArrayType>(T))
    return isa<ArrayType>(ArrType->getElementType());
  return false;
}

unsigned DXILFlattenArraysVisitor::getTotalElements(Type *ArrayTy) {
  unsigned TotalElements = 1;
  Type *CurrArrayTy = ArrayTy;
  while (auto *InnerArrayTy = dyn_cast<ArrayType>(CurrArrayTy)) {
    TotalElements *= InnerArrayTy->getNumElements();
    CurrArrayTy = InnerArrayTy->getElementType();
  }
  return TotalElements;
}

Type *DXILFlattenArraysVisitor::getBaseElementType(Type *ArrayTy) {
  Type *CurrArrayTy = ArrayTy;
  while (auto *InnerArrayTy = dyn_cast<ArrayType>(CurrArrayTy)) {
    CurrArrayTy = InnerArrayTy->getElementType();
  }
  return CurrArrayTy;
}

ConstantInt *
DXILFlattenArraysVisitor::flattenIndices(ArrayRef<ConstantInt *> Indices,
                                         ArrayRef<uint64_t> Dims,
                                         IRBuilder<> &Builder) {
  assert(Indices.size() == Dims.size() &&
         "Indicies and dimmensions should be the same");
  unsigned FlatIndex = 0;
  unsigned Multiplier = 1;

  for (int I = Indices.size() - 1; I >= 0; --I) {
    unsigned DimSize = Dims[I];
    FlatIndex += Indices[I]->getZExtValue() * Multiplier;
    Multiplier *= DimSize;
  }
  return Builder.getInt32(FlatIndex);
}

bool DXILFlattenArraysVisitor::visitAllocaInst(AllocaInst &AI) {
  if (!isMultiDimensionalArray(AI.getAllocatedType()))
    return false;

  ArrayType *ArrType = cast<ArrayType>(AI.getAllocatedType());
  IRBuilder<> Builder(&AI);
  unsigned TotalElements = getTotalElements(ArrType);

  ArrayType *FattenedArrayType =
      ArrayType::get(getBaseElementType(ArrType), TotalElements);
  AllocaInst *FlatAlloca =
      Builder.CreateAlloca(FattenedArrayType, nullptr, AI.getName() + ".flat");
  FlatAlloca->setAlignment(AI.getAlign());
  AI.replaceAllUsesWith(FlatAlloca);
  AI.eraseFromParent();
  return true;
}

void DXILFlattenArraysVisitor::recursivelyCollectGEPs(
    GetElementPtrInst &CurrGEP, GetElementPtrInst &NewGEP,
    GEPChainInfo &GEPChain, SmallVector<ConstantInt *> Indices,
    SmallVector<uint64_t> Dims) {
  ConstantInt *LastIndex =
      cast<ConstantInt>(CurrGEP.getOperand(CurrGEP.getNumOperands() - 1));

  Indices.push_back(LastIndex);
  assert(isa<ArrayType>(CurrGEP.getSourceElementType()));
  Dims.push_back(
      cast<ArrayType>(CurrGEP.getSourceElementType())->getNumElements());
  if (!isMultiDimensionalArray(CurrGEP.getSourceElementType())) {
    GEPChain.GEPToIndicesMap.insert({&CurrGEP, Indices});
    GEPChain.GEPChildToNewParentMap.insert({&CurrGEP, &NewGEP});
    GEPChain.GEPToDimmsMap.insert({&CurrGEP, Dims});
  }
  for (auto *User : CurrGEP.users()) {
    if (GetElementPtrInst *NestedGEP = dyn_cast<GetElementPtrInst>(User)) {
      recursivelyCollectGEPs(*NestedGEP, NewGEP, GEPChain, Indices, Dims);
    }
  }
  PotentiallyDeadInstrs.emplace_back(&CurrGEP);
}

bool DXILFlattenArraysVisitor::visitGetElementPtrInstInGEPChain(
    GetElementPtrInst &GEP) {
  IRBuilder<> Builder(&GEP);
  SmallVector<ConstantInt *> Indices = GEPChain.GEPToIndicesMap.at(&GEP);
  GetElementPtrInst *Parent = GEPChain.GEPChildToNewParentMap.at(&GEP);
  SmallVector<uint64_t> Dims = GEPChain.GEPToDimmsMap.at(&GEP);
  ConstantInt *FlatIndex = flattenIndices(Indices, Dims, Builder);
  if (!FlatIndex->isZero()) {
    ArrayType *FlattenedArrayType =
        cast<ArrayType>(Parent->getSourceElementType());
    Value *FlatGEP =
        Builder.CreateGEP(FlattenedArrayType, Parent->getPointerOperand(),
                          FlatIndex, GEP.getName() + ".flat", GEP.isInBounds());

    GEP.replaceAllUsesWith(FlatGEP);
  }
  GEP.eraseFromParent();
  return true;
}

bool DXILFlattenArraysVisitor::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  auto It = GEPChain.GEPToIndicesMap.find(&GEP);
  if (It != GEPChain.GEPToIndicesMap.end())
    return visitGetElementPtrInstInGEPChain(GEP);
  if (!isMultiDimensionalArray(GEP.getSourceElementType()))
    return false;

  ArrayType *ArrType = cast<ArrayType>(GEP.getSourceElementType());
  IRBuilder<> Builder(&GEP);
  unsigned TotalElements = getTotalElements(ArrType);
  ArrayType *FattenedArrayType =
      ArrayType::get(getBaseElementType(ArrType), TotalElements);

  ConstantInt *FlatIndex = Builder.getInt32(0);
  Value *PtrOperand = GEP.getPointerOperand();
  auto OpExists = GEPChain.OperandToBaseGEPMap.find(PtrOperand);

  GetElementPtrInst *FlatGEP = nullptr;
  if (OpExists == GEPChain.OperandToBaseGEPMap.end()) {
    FlatGEP = cast<GetElementPtrInst>(
        Builder.CreateGEP(FattenedArrayType, PtrOperand, FlatIndex,
                          GEP.getName() + ".flat", GEP.isInBounds()));
    GEPChain.OperandToBaseGEPMap.insert({PtrOperand, FlatGEP});
  } else
    FlatGEP = OpExists->getSecond();
  recursivelyCollectGEPs(GEP, *FlatGEP, GEPChain);
  GEP.replaceAllUsesWith(FlatGEP);
  GEP.eraseFromParent();
  return true;
}

bool DXILFlattenArraysVisitor::visit(Function &F) {
  bool MadeChange = false;
  ////for (BasicBlock &BB : make_early_inc_range(F)) {
  ReversePostOrderTraversal<Function *> RPOT(&F);
  for (BasicBlock *BB : make_early_inc_range(RPOT)) {
    for (Instruction &I : make_early_inc_range(*BB)) {
      if (InstVisitor::visit(I) && I.getType()->isVoidTy()) {
        I.eraseFromParent();
        MadeChange = true;
      }
    }
  }
  return MadeChange;
}

static bool flattenArrays(Module &M) {
  // TODO
  bool MadeChange = false;
  DXILFlattenArraysVisitor Impl;
  for (auto &F : make_early_inc_range(M.functions())) {
    MadeChange = Impl.visit(F);
  }
  return MadeChange;
}

PreservedAnalyses DXILFlattenArrays::run(Module &M, ModuleAnalysisManager &) {
  bool MadeChanges = flattenArrays(M);
  if (!MadeChanges)
    return PreservedAnalyses::all();
  PreservedAnalyses PA;
  PA.preserve<DXILResourceAnalysis>();
  return PA;
}

bool DXILFlattenArraysLegacy::runOnModule(Module &M) {
  return flattenArrays(M);
}

void DXILFlattenArraysLegacy::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addPreserved<DXILResourceWrapperPass>();
}

char DXILFlattenArraysLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(DXILFlattenArraysLegacy, DEBUG_TYPE,
                      "DXIL Array Flattener", false, false)
INITIALIZE_PASS_END(DXILFlattenArraysLegacy, DEBUG_TYPE, "DXIL Array Flattener",
                    false, false)

ModulePass *llvm::createDXILFlattenArraysLegacyPass() {
  return new DXILFlattenArraysLegacy();
}