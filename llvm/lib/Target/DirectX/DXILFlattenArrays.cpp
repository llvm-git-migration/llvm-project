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
#include <cstddef>
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

struct GEPData {
  ArrayType *ParentArrayType;
  Value *ParendOperand;
  SmallVector<Value *> Indices;
  SmallVector<uint64_t> Dims;
  bool AllIndicesAreConstInt;
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
  static bool isMultiDimensionalArray(Type *T);
  static unsigned getTotalElements(Type *ArrayTy);
  static Type *getBaseElementType(Type *ArrayTy);

private:
  SmallVector<WeakTrackingVH, 32> PotentiallyDeadInstrs;
  DenseMap<GetElementPtrInst *, GEPData> GEPChainMap;
  bool finish();
  ConstantInt *constFlattenIndices(ArrayRef<Value *> Indices,
                                   ArrayRef<uint64_t> Dims,
                                   IRBuilder<> &Builder);
  Value *instructionFlattenIndices(ArrayRef<Value *> Indices,
                                   ArrayRef<uint64_t> Dims,
                                   IRBuilder<> &Builder);
  void
  recursivelyCollectGEPs(GetElementPtrInst &CurrGEP,
                         ArrayType *FlattenedArrayType, Value *PtrOperand,
                         unsigned &UseCount,
                         SmallVector<Value *> Indices = SmallVector<Value *>(),
                         SmallVector<uint64_t> Dims = SmallVector<uint64_t>(),
                         bool AllIndicesAreConstInt = true);
  ConstantInt *computeFlatIndex(GetElementPtrInst &GEP);
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

ConstantInt *DXILFlattenArraysVisitor::constFlattenIndices(
    ArrayRef<Value *> Indices, ArrayRef<uint64_t> Dims, IRBuilder<> &Builder) {
  assert(Indices.size() == Dims.size() &&
         "Indicies and dimmensions should be the same");
  unsigned FlatIndex = 0;
  unsigned Multiplier = 1;

  for (int I = Indices.size() - 1; I >= 0; --I) {
    unsigned DimSize = Dims[I];
    ConstantInt *CIndex = dyn_cast<ConstantInt>(Indices[I]);
    assert(CIndex && "This function expects all indicies to be ConstantInt");
    FlatIndex += CIndex->getZExtValue() * Multiplier;
    Multiplier *= DimSize;
  }
  return Builder.getInt32(FlatIndex);
}

Value *DXILFlattenArraysVisitor::instructionFlattenIndices(
    ArrayRef<Value *> Indices, ArrayRef<uint64_t> Dims, IRBuilder<> &Builder) {
  if (Indices.size() == 1)
    return Indices[0];

  Value *FlatIndex = Builder.getInt32(0);
  unsigned Multiplier = 1;

  for (int I = Indices.size() - 1; I >= 0; --I) {
    unsigned DimSize = Dims[I];
    Value *VMultiplier = Builder.getInt32(Multiplier);
    Value *ScaledIndex = Builder.CreateMul(Indices[I], VMultiplier);
    FlatIndex = Builder.CreateAdd(FlatIndex, ScaledIndex);
    Multiplier *= DimSize;
  }
  return FlatIndex;
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

ConstantInt *
DXILFlattenArraysVisitor::computeFlatIndex(GetElementPtrInst &GEP) {
  unsigned IndexAmount = GEP.getNumIndices();
  assert(IndexAmount >= 1 && "Need At least one Index");
  if (IndexAmount == 1)
    return dyn_cast<ConstantInt>(GEP.getOperand(GEP.getNumOperands() - 1));

  // Get the type of the base pointer.
  Type *BaseType = GEP.getSourceElementType();

  // Determine the dimensions of the multi-dimensional array.
  SmallVector<int64_t> Dimensions;
  while (auto *ArrType = dyn_cast<ArrayType>(BaseType)) {
    Dimensions.push_back(ArrType->getNumElements());
    BaseType = ArrType->getElementType();
  }
  unsigned FlatIndex = 0;
  unsigned Multiplier = 1;
  unsigned BitWidth = 32;
  for (const Use &Index : GEP.indices()) {
    ConstantInt *CurrentIndex = dyn_cast<ConstantInt>(Index);
    BitWidth = CurrentIndex->getBitWidth();
    if (!CurrentIndex)
      return nullptr;
    int64_t IndexValue = CurrentIndex->getSExtValue();
    FlatIndex += IndexValue * Multiplier;

    if (!Dimensions.empty()) {
      Multiplier *= Dimensions.back(); // Use the last dimension size
      Dimensions.pop_back();           // Remove the last dimension
    }
  }
  return ConstantInt::get(GEP.getContext(), APInt(BitWidth, FlatIndex));
}

void DXILFlattenArraysVisitor::recursivelyCollectGEPs(
    GetElementPtrInst &CurrGEP, ArrayType *FlattenedArrayType,
    Value *PtrOperand, unsigned &UseCount, SmallVector<Value *> Indices,
    SmallVector<uint64_t> Dims, bool AllIndicesAreConstInt) {
  Value *LastIndex = CurrGEP.getOperand(CurrGEP.getNumOperands() - 1);
  AllIndicesAreConstInt &= isa<ConstantInt>(LastIndex);
  Indices.push_back(LastIndex);
  assert(isa<ArrayType>(CurrGEP.getSourceElementType()));
  Dims.push_back(
      cast<ArrayType>(CurrGEP.getSourceElementType())->getNumElements());
  if (!isMultiDimensionalArray(CurrGEP.getSourceElementType())) {
    GEPChainMap.insert(
        {&CurrGEP,
         {std::move(FlattenedArrayType), PtrOperand, std::move(Indices),
          std::move(Dims), AllIndicesAreConstInt}});
  }
  for (auto *User : CurrGEP.users()) {
    if (GetElementPtrInst *NestedGEP = dyn_cast<GetElementPtrInst>(User)) {
      recursivelyCollectGEPs(*NestedGEP, FlattenedArrayType, PtrOperand,
                             ++UseCount, Indices, Dims, AllIndicesAreConstInt);
    }
  }
  assert(Dims.size() == Indices.size());
  // If the std::moves did not happen the gep chain is incomplete
  // let save the last state.
  if (!Dims.empty())
    GEPChainMap.insert(
        {&CurrGEP,
         {std::move(FlattenedArrayType), PtrOperand, std::move(Indices),
          std::move(Dims), AllIndicesAreConstInt}});
}

bool DXILFlattenArraysVisitor::visitGetElementPtrInstInGEPChain(
    GetElementPtrInst &GEP) {
  IRBuilder<> Builder(&GEP);
  GEPData GEPInfo = GEPChainMap.at(&GEP);
  Value *FlatIndex;
  if (GEPInfo.AllIndicesAreConstInt)
    FlatIndex = constFlattenIndices(GEPInfo.Indices, GEPInfo.Dims, Builder);
  else
    FlatIndex =
        instructionFlattenIndices(GEPInfo.Indices, GEPInfo.Dims, Builder);

  ArrayType *FlattenedArrayType = GEPInfo.ParentArrayType;
  Value *FlatGEP =
      Builder.CreateGEP(FlattenedArrayType, GEPInfo.ParendOperand, FlatIndex,
                        GEP.getName() + ".flat", GEP.isInBounds());

  GEP.replaceAllUsesWith(FlatGEP);
  GEP.eraseFromParent();
  return true;
}

bool DXILFlattenArraysVisitor::visitGetElementPtrInst(GetElementPtrInst &GEP) {
  auto It = GEPChainMap.find(&GEP);
  if (It != GEPChainMap.end())
    return visitGetElementPtrInstInGEPChain(GEP);
  if (!isMultiDimensionalArray(GEP.getSourceElementType()))
    return false;

  ArrayType *ArrType = cast<ArrayType>(GEP.getSourceElementType());
  IRBuilder<> Builder(&GEP);
  unsigned TotalElements = getTotalElements(ArrType);
  ArrayType *FlattenedArrayType =
      ArrayType::get(getBaseElementType(ArrType), TotalElements);

  Value *PtrOperand = GEP.getPointerOperand();
  // if(isa<ConstantInt>(GEP.getOperand(GEP.getNumOperands() - 1))) {
  unsigned UseCount = 0;
  recursivelyCollectGEPs(GEP, FlattenedArrayType, PtrOperand, UseCount);
  if (UseCount == 0)
    visitGetElementPtrInstInGEPChain(GEP);
  else
    PotentiallyDeadInstrs.emplace_back(&GEP);
  /*} else {
    SmallVector<Value *> Indices(GEP.idx_begin(),GEP.idx_end());
     Value *FlatGEP =
      Builder.CreateGEP(FlattenedArrayType, PtrOperand, Indices,
                        GEP.getName() + ".flat", GEP.isInBounds());
  GEP.replaceAllUsesWith(FlatGEP);
  GEP.eraseFromParent();
  }*/
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
  finish();
  return MadeChange;
}

static void collectElements(Constant *Init,
                            SmallVectorImpl<Constant *> &Elements) {
  // Base case: If Init is not an array, add it directly to the vector.
  if (!isa<ArrayType>(Init->getType())) {
    Elements.push_back(Init);
    return;
  }

  // Recursive case: Process each element in the array.
  if (auto *ArrayConstant = dyn_cast<ConstantArray>(Init)) {
    for (unsigned I = 0; I < ArrayConstant->getNumOperands(); ++I) {
      collectElements(ArrayConstant->getOperand(I), Elements);
    }
  } else if (auto *DataArrayConstant = dyn_cast<ConstantDataArray>(Init)) {
    for (unsigned I = 0; I < DataArrayConstant->getNumElements(); ++I) {
      collectElements(DataArrayConstant->getElementAsConstant(I), Elements);
    }
  } else {
    assert(
        false &&
        "Expected a ConstantArray or ConstantDataArray for array initializer!");
  }
}

static Constant *transformInitializer(Constant *Init, Type *OrigType,
                                      ArrayType *FlattenedType,
                                      LLVMContext &Ctx) {
  // Handle ConstantAggregateZero (zero-initialized constants)
  if (isa<ConstantAggregateZero>(Init))
    return ConstantAggregateZero::get(FlattenedType);

  // Handle UndefValue (undefined constants)
  if (isa<UndefValue>(Init))
    return UndefValue::get(FlattenedType);

  if (!isa<ArrayType>(OrigType))
    return Init;

  SmallVector<Constant *> FlattenedElements;
  collectElements(Init, FlattenedElements);
  assert(FlattenedType->getNumElements() == FlattenedElements.size() &&
         "The number of collected elements should match the FlattenedType");
  return ConstantArray::get(FlattenedType, FlattenedElements);
}

static void
flattenGlobalArrays(Module &M,
                    DenseMap<GlobalVariable *, GlobalVariable *> &GlobalMap) {
  LLVMContext &Ctx = M.getContext();
  for (GlobalVariable &G : M.globals()) {
    Type *OrigType = G.getValueType();
    if (!DXILFlattenArraysVisitor::isMultiDimensionalArray(OrigType))
      continue;

    ArrayType *ArrType = cast<ArrayType>(OrigType);
    unsigned TotalElements =
        DXILFlattenArraysVisitor::getTotalElements(ArrType);
    ArrayType *FattenedArrayType = ArrayType::get(
        DXILFlattenArraysVisitor::getBaseElementType(ArrType), TotalElements);

    // Create a new global variable with the updated type
    // Note: Initializer is set via transformInitializer
    GlobalVariable *NewGlobal =
        new GlobalVariable(M, FattenedArrayType, G.isConstant(), G.getLinkage(),
                           /*Initializer=*/nullptr, G.getName() + ".1dim", &G,
                           G.getThreadLocalMode(), G.getAddressSpace(),
                           G.isExternallyInitialized());

    // Copy relevant attributes
    NewGlobal->setUnnamedAddr(G.getUnnamedAddr());
    if (G.getAlignment() > 0) {
      NewGlobal->setAlignment(G.getAlign());
    }

    if (G.hasInitializer()) {
      Constant *Init = G.getInitializer();
      Constant *NewInit =
          transformInitializer(Init, OrigType, FattenedArrayType, Ctx);
      NewGlobal->setInitializer(NewInit);
    }
    GlobalMap[&G] = NewGlobal;
  }
}

static bool flattenArrays(Module &M) {
  bool MadeChange = false;
  DXILFlattenArraysVisitor Impl;
  DenseMap<GlobalVariable *, GlobalVariable *> GlobalMap;
  flattenGlobalArrays(M, GlobalMap);
  for (auto &F : make_early_inc_range(M.functions())) {
    if (F.isIntrinsic())
      continue;
    MadeChange |= Impl.visit(F);
  }
  for (auto &[Old, New] : GlobalMap) {
    Old->replaceAllUsesWith(New);
    Old->eraseFromParent();
    MadeChange |= true;
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
