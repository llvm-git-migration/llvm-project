#include "AMDGPUAsanInstrumentation.h"

#define DEBUG_TYPE "amdgpu-asan-instrumentation"

using namespace llvm;

namespace llvm {
namespace AMDGPU {

const char kAMDGPUBallotName[] = "llvm.amdgcn.ballot.i64";
const char kAMDGPUUnreachableName[] = "llvm.amdgcn.unreachable";
const char kAMDGPULDSKernelId[] = "llvm.amdgcn.lds.kernel.id";

static const uint64_t kSmallX86_64ShadowOffsetBase = 0x7FFFFFFF;
static const uint64_t kSmallX86_64ShadowOffsetAlignMask = ~0xFFFULL;

static uint64_t getRedzoneSizeForScale(int AsanScale) {
  // Redzone used for stack and globals is at least 32 bytes.
  // For scales 6 and 7, the redzone has to be 64 and 128 bytes respectively.
  return std::max(32U, 1U << AsanScale);
}

static uint64_t getMinRedzoneSizeForGlobal(int AsanScale) {
  return getRedzoneSizeForScale(AsanScale);
}

uint64_t getRedzoneSizeForGlobal(int AsanScale, uint64_t SizeInBytes) {
  constexpr uint64_t kMaxRZ = 1 << 18;
  const uint64_t MinRZ = getMinRedzoneSizeForGlobal(AsanScale);

  uint64_t RZ = 0;
  if (SizeInBytes <= MinRZ / 2) {
    // Reduce redzone size for small size objects, e.g. int, char[1]. MinRZ is
    // at least 32 bytes, optimize when SizeInBytes is less than or equal to
    // half of MinRZ.
    RZ = MinRZ - SizeInBytes;
  } else {
    // Calculate RZ, where MinRZ <= RZ <= MaxRZ, and RZ ~ 1/4 * SizeInBytes.
    RZ = std::clamp((SizeInBytes / MinRZ / 4) * MinRZ, MinRZ, kMaxRZ);

    // Round up to multiple of MinRZ.
    if (SizeInBytes % MinRZ)
      RZ += MinRZ - (SizeInBytes % MinRZ);
  }

  assert((RZ + SizeInBytes) % MinRZ == 0);

  return RZ;
}

static size_t TypeStoreSizeToSizeIndex(uint32_t TypeSize) {
  size_t Res = llvm::countr_zero(TypeSize / 8);
  return Res;
}

static Instruction *genAMDGPUReportBlock(Module &M, IRBuilder<> &IRB,
                                         Value *Cond, bool Recover) {
  Value *ReportCond = Cond;
  if (!Recover) {
    auto Ballot = M.getOrInsertFunction(kAMDGPUBallotName, IRB.getInt64Ty(),
                                        IRB.getInt1Ty());
    ReportCond = IRB.CreateIsNotNull(IRB.CreateCall(Ballot, {Cond}));
  }

  auto *Trm = SplitBlockAndInsertIfThen(
      ReportCond, &*IRB.GetInsertPoint(), false,
      MDBuilder(M.getContext()).createBranchWeights(1, 100000));
  Trm->getParent()->setName("asan.report");

  if (Recover)
    return Trm;

  Trm = SplitBlockAndInsertIfThen(Cond, Trm, false);
  IRB.SetInsertPoint(Trm);
  return IRB.CreateCall(
      M.getOrInsertFunction(kAMDGPUUnreachableName, IRB.getVoidTy()), {});
}

static Value *createSlowPathCmp(Module &M, IRBuilder<> &IRB, Value *AddrLong,
                                Value *ShadowValue, uint32_t TypeStoreSize,
                                int AsanScale) {

  unsigned int LongSize = M.getDataLayout().getPointerSizeInBits();
  IntegerType *IntptrTy = Type::getIntNTy(M.getContext(), LongSize);
  size_t Granularity = static_cast<size_t>(1) << AsanScale;
  // Addr & (Granularity - 1)
  Value *LastAccessedByte =
      IRB.CreateAnd(AddrLong, ConstantInt::get(IntptrTy, Granularity - 1));
  // (Addr & (Granularity - 1)) + size - 1
  if (TypeStoreSize / 8 > 1)
    LastAccessedByte = IRB.CreateAdd(
        LastAccessedByte, ConstantInt::get(IntptrTy, TypeStoreSize / 8 - 1));
  // (uint8_t) ((Addr & (Granularity-1)) + size - 1)
  LastAccessedByte =
      IRB.CreateIntCast(LastAccessedByte, ShadowValue->getType(), false);
  // ((uint8_t) ((Addr & (Granularity-1)) + size - 1)) >= ShadowValue
  return IRB.CreateICmpSGE(LastAccessedByte, ShadowValue);
}

static Instruction *generateCrashCode(Module &M, IRBuilder<> &IRB,
                                      Instruction *InsertBefore, Value *Addr,
                                      bool IsWrite, size_t AccessSizeIndex,
                                      Value *SizeArgument, bool Recover) {
  IRB.SetInsertPoint(InsertBefore);
  CallInst *Call = nullptr;
  int LongSize = M.getDataLayout().getPointerSizeInBits();
  Type *IntptrTy = Type::getIntNTy(M.getContext(), LongSize);
  const char kAsanReportErrorTemplate[] = "__asan_report_";
  const std::string TypeStr = IsWrite ? "store" : "load";
  const std::string EndingStr = Recover ? "_noabort" : "";
  SmallVector<Type *, 3> Args2 = {IntptrTy, IntptrTy};
  AttributeList AL2;
  FunctionCallee AsanErrorCallbackSized = M.getOrInsertFunction(
      kAsanReportErrorTemplate + TypeStr + "_n" + EndingStr,
      FunctionType::get(IRB.getVoidTy(), Args2, false), AL2);
  const std::string Suffix = TypeStr + llvm::itostr(1ULL << AccessSizeIndex);
  SmallVector<Type *, 2> Args1{1, IntptrTy};
  AttributeList AL1;
  FunctionCallee AsanErrorCallback = M.getOrInsertFunction(
      kAsanReportErrorTemplate + Suffix + EndingStr,
      FunctionType::get(IRB.getVoidTy(), Args1, false), AL1);
  if (SizeArgument) {
    Call = IRB.CreateCall(AsanErrorCallbackSized, {Addr, SizeArgument});
  } else {
    Call = IRB.CreateCall(AsanErrorCallback, Addr);
  }

  Call->setCannotMerge();
  return Call;
}

static Value *memToShadow(Module &M, IRBuilder<> &IRB, Value *Shadow,
                          int AsanScale, uint32_t AsanOffset) {
  int LongSize = M.getDataLayout().getPointerSizeInBits();
  Type *IntptrTy = Type::getIntNTy(M.getContext(), LongSize);
  // Shadow >> scale
  Shadow = IRB.CreateLShr(Shadow, AsanScale);
  if (AsanOffset == 0)
    return Shadow;
  // (Shadow >> scale) | offset
  Value *ShadowBase = ConstantInt::get(IntptrTy, AsanOffset);
  return IRB.CreateAdd(Shadow, ShadowBase);
}

void instrumentAddress(Module &M, IRBuilder<> &IRB, Instruction *OrigIns,
                       Instruction *InsertBefore, Value *Addr,
                       MaybeAlign Alignment, uint32_t TypeStoreSize,
                       bool IsWrite, Value *SizeArgument, bool UseCalls,
                       bool Recover, int AsanScale, int AsanOffset) {
  int LongSize = M.getDataLayout().getPointerSizeInBits();
  Type *IntptrTy = Type::getIntNTy(M.getContext(), LongSize);
  IRB.SetInsertPoint(InsertBefore);
  size_t AccessSizeIndex = TypeStoreSizeToSizeIndex(TypeStoreSize);
  Type *ShadowTy = IntegerType::get(M.getContext(),
                                    std::max(8U, TypeStoreSize >> AsanScale));
  Type *ShadowPtrTy = PointerType::get(ShadowTy, 0);
  Value *AddrLong = IRB.CreatePointerCast(Addr, IntptrTy);
  Value *ShadowPtr = memToShadow(M, IRB, AddrLong, AsanScale, AsanOffset);
  const uint64_t ShadowAlign =
      std::max<uint64_t>(Alignment.valueOrOne().value() >> AsanScale, 1);
  Value *ShadowValue = IRB.CreateAlignedLoad(
      ShadowTy, IRB.CreateIntToPtr(ShadowPtr, ShadowPtrTy), Align(ShadowAlign));
  Value *Cmp = IRB.CreateIsNotNull(ShadowValue);
  auto *Cmp2 = createSlowPathCmp(M, IRB, AddrLong, ShadowValue, TypeStoreSize,
                                 AsanScale);
  Cmp = IRB.CreateAnd(Cmp, Cmp2);
  Instruction *CrashTerm = genAMDGPUReportBlock(M, IRB, Cmp, Recover);
  Instruction *Crash =
      generateCrashCode(M, IRB, CrashTerm, AddrLong, IsWrite, AccessSizeIndex,
                        SizeArgument, Recover);
  if (OrigIns->getDebugLoc())
    Crash->setDebugLoc(OrigIns->getDebugLoc());
  return;
}

void getInterestingMemoryOperands(
    Module &M, Instruction *I,
    SmallVectorImpl<InterestingMemoryOperand> &Interesting) {
  const DataLayout &DL = M.getDataLayout();
  unsigned int LongSize = M.getDataLayout().getPointerSizeInBits();
  Type *IntptrTy = Type::getIntNTy(M.getContext(), LongSize);
  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    Interesting.emplace_back(I, LI->getPointerOperandIndex(), false,
                             LI->getType(), LI->getAlign());
  } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    Interesting.emplace_back(I, SI->getPointerOperandIndex(), true,
                             SI->getValueOperand()->getType(), SI->getAlign());
  } else if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(I)) {
    Interesting.emplace_back(I, RMW->getPointerOperandIndex(), true,
                             RMW->getValOperand()->getType(), std::nullopt);
  } else if (AtomicCmpXchgInst *XCHG = dyn_cast<AtomicCmpXchgInst>(I)) {
    Interesting.emplace_back(I, XCHG->getPointerOperandIndex(), true,
                             XCHG->getCompareOperand()->getType(),
                             std::nullopt);
  } else if (auto CI = dyn_cast<CallInst>(I)) {
    switch (CI->getIntrinsicID()) {
    case Intrinsic::masked_load:
    case Intrinsic::masked_store:
    case Intrinsic::masked_gather:
    case Intrinsic::masked_scatter: {
      bool IsWrite = CI->getType()->isVoidTy();
      // Masked store has an initial operand for the value.
      unsigned OpOffset = IsWrite ? 1 : 0;
      Type *Ty = IsWrite ? CI->getArgOperand(0)->getType() : CI->getType();
      MaybeAlign Alignment = Align(1);
      // Otherwise no alignment guarantees. We probably got Undef.
      if (auto *Op = dyn_cast<ConstantInt>(CI->getOperand(1 + OpOffset)))
        Alignment = Op->getMaybeAlignValue();
      Value *Mask = CI->getOperand(2 + OpOffset);
      Interesting.emplace_back(I, OpOffset, IsWrite, Ty, Alignment, Mask);
      break;
    }
    case Intrinsic::masked_expandload:
    case Intrinsic::masked_compressstore: {
      bool IsWrite = CI->getIntrinsicID() == Intrinsic::masked_compressstore;
      unsigned OpOffset = IsWrite ? 1 : 0;
      auto BasePtr = CI->getOperand(OpOffset);
      MaybeAlign Alignment = BasePtr->getPointerAlignment(DL);
      Type *Ty = IsWrite ? CI->getArgOperand(0)->getType() : CI->getType();
      IRBuilder<> IB(I);
      Value *Mask = CI->getOperand(1 + OpOffset);
      // Use the popcount of Mask as the effective vector length.
      Type *ExtTy = VectorType::get(IntptrTy, cast<VectorType>(Ty));
      Value *ExtMask = IB.CreateZExt(Mask, ExtTy);
      Value *EVL = IB.CreateAddReduce(ExtMask);
      Value *TrueMask = ConstantInt::get(Mask->getType(), 1);
      Interesting.emplace_back(I, OpOffset, IsWrite, Ty, Alignment, TrueMask,
                               EVL);
      break;
    }
    case Intrinsic::vp_load:
    case Intrinsic::vp_store:
    case Intrinsic::experimental_vp_strided_load:
    case Intrinsic::experimental_vp_strided_store: {
      auto *VPI = cast<VPIntrinsic>(CI);
      unsigned IID = CI->getIntrinsicID();
      bool IsWrite = CI->getType()->isVoidTy();
      unsigned PtrOpNo = *VPI->getMemoryPointerParamPos(IID);
      Type *Ty = IsWrite ? CI->getArgOperand(0)->getType() : CI->getType();
      MaybeAlign Alignment = VPI->getOperand(PtrOpNo)->getPointerAlignment(DL);
      Value *Stride = nullptr;
      if (IID == Intrinsic::experimental_vp_strided_store ||
          IID == Intrinsic::experimental_vp_strided_load) {
        Stride = VPI->getOperand(PtrOpNo + 1);
        // Use the pointer alignment as the element alignment if the stride is a
        // mutiple of the pointer alignment. Otherwise, the element alignment
        // should be Align(1).
        unsigned PointerAlign = Alignment.valueOrOne().value();
        if (!isa<ConstantInt>(Stride) ||
            cast<ConstantInt>(Stride)->getZExtValue() % PointerAlign != 0)
          Alignment = Align(1);
      }
      Interesting.emplace_back(I, PtrOpNo, IsWrite, Ty, Alignment,
                               VPI->getMaskParam(), VPI->getVectorLengthParam(),
                               Stride);
      break;
    }
    case Intrinsic::vp_gather:
    case Intrinsic::vp_scatter: {
      auto *VPI = cast<VPIntrinsic>(CI);
      unsigned IID = CI->getIntrinsicID();
      bool IsWrite = IID == Intrinsic::vp_scatter;
      unsigned PtrOpNo = *VPI->getMemoryPointerParamPos(IID);
      Type *Ty = IsWrite ? CI->getArgOperand(0)->getType() : CI->getType();
      MaybeAlign Alignment = VPI->getPointerAlignment();
      Interesting.emplace_back(I, PtrOpNo, IsWrite, Ty, Alignment,
                               VPI->getMaskParam(),
                               VPI->getVectorLengthParam());
      break;
    }
    default:
      for (unsigned ArgNo = 0; ArgNo < CI->arg_size(); ArgNo++) {
        if (!CI->isByValArgument(ArgNo))
          continue;
        Type *Ty = CI->getParamByValType(ArgNo);
        Interesting.emplace_back(I, ArgNo, false, Ty, Align(1));
      }
    }
  }
}
} // end namespace AMDGPU
} // end namespace llvm
