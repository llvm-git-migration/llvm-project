//===--------------- RISCVPostLegalizerLowering.cpp -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Post-legalization lowering for instructions.
///
/// This is used to offload pattern matching from the selector.
///
/// General optimization combines should be handled by either the
/// RISCVPostLegalizerCombiner or the RISCVPreLegalizerCombiner.
///
//===----------------------------------------------------------------------===//

#include "RISCVSubtarget.h"

#include "llvm/CodeGen/GlobalISel/Combiner.h"
#include "llvm/CodeGen/GlobalISel/CombinerHelper.h"
#include "llvm/CodeGen/GlobalISel/GIMatchTableExecutorImpl.h"
#include "llvm/CodeGen/GlobalISel/GISelChangeObserver.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

#define GET_GICOMBINER_DEPS
#include "RISCVGenPostLegalizeGILowering.inc"
#undef GET_GICOMBINER_DEPS

#define DEBUG_TYPE "riscv-postlegalizer-lowering"

using namespace llvm;

namespace {

#define GET_GICOMBINER_TYPES
#include "RISCVGenPostLegalizeGILowering.inc"
#undef GET_GICOMBINER_TYPES

static LLT getLMUL1Ty(LLT VecTy) {
  assert(VecTy.getElementType().getSizeInBits() <= 64 &&
         "Unexpected vector LLT");
  return LLT::scalable_vector(RISCV::RVVBitsPerBlock /
                                  VecTy.getElementType().getSizeInBits(),
                              VecTy.getElementType());
}

/// Return the type of the mask type suitable for masking the provided
/// vector type.  This is simply an i1 element type vector of the same
/// (possibly scalable) length.
static LLT getMaskTypeFor(LLT VecTy) {
  assert(VecTy.isVector());
  ElementCount EC = VecTy.getElementCount();
  return LLT::vector(EC, LLT::scalar(1));
}

/// Creates an all ones mask suitable for masking a vector of type VecTy with
/// vector length VL.
static MachineInstrBuilder buildAllOnesMask(LLT VecTy, const SrcOp &VL,
                                            MachineIRBuilder &MIB,
                                            MachineRegisterInfo &MRI) {
  LLT MaskTy = getMaskTypeFor(VecTy);
  return MIB.buildInstr(RISCV::G_VMSET_VL, {MaskTy}, {VL});
}

/// Gets the two common "VL" operands: an all-ones mask and the vector length.
/// VecTy is a scalable vector type.
static std::pair<MachineInstrBuilder, Register>
buildDefaultVLOps(const DstOp &Dst, MachineIRBuilder &MIB,
                  MachineRegisterInfo &MRI) {
  LLT VecTy = Dst.getLLTTy(MRI);
  assert(VecTy.isScalableVector() && "Expecting scalable container type");
  Register VL(RISCV::X0);
  MachineInstrBuilder Mask = buildAllOnesMask(VecTy, VL, MIB, MRI);
  return {Mask, VL};
}

/// Lowers G_INSERT_SUBVECTOR. We know we can lower it here since the legalizer
/// marked it as legal.
void lowerInsertSubvector(MachineInstr &MI, const RISCVSubtarget &STI) {
  GInsertSubvector &IS = cast<GInsertSubvector>(MI);

  MachineIRBuilder MIB(MI);
  MachineRegisterInfo &MRI = *MIB.getMRI();

  Register Dst = IS.getReg(0);
  Register Src1 = IS.getBigVec();
  Register Src2 = IS.getSubVec();
  uint64_t Idx = IS.getIndexImm();

  LLT BigTy = MRI.getType(Src1);
  LLT LitTy = MRI.getType(Src2);
  Register BigVec = Src1;
  Register LitVec = Src2;

  // We don't have the ability to slide mask vectors up indexed by their i1
  // elements; the smallest we can do is i8. Often we are able to bitcast to
  // equivalent i8 vectors. Otherwise, we can must zeroextend to equivalent i8
  // vectors and truncate down after the insert.
  if (LitTy.getElementType() == LLT::scalar(1) &&
      (Idx != 0 ||
       MRI.getVRegDef(BigVec)->getOpcode() != TargetOpcode::G_IMPLICIT_DEF)) {
    auto BigTyMinElts = BigTy.getElementCount().getKnownMinValue();
    auto LitTyMinElts = LitTy.getElementCount().getKnownMinValue();
    if (BigTyMinElts >= 8 && LitTyMinElts >= 8) {
      assert(Idx % 8 == 0 && "Invalid index");
      assert(BigTyMinElts % 8 == 0 && LitTyMinElts % 8 == 0 &&
             "Unexpected mask vector lowering");
      Idx /= 8;
      BigTy = LLT::vector(BigTy.getElementCount().divideCoefficientBy(8), 8);
      LitTy = LLT::vector(LitTy.getElementCount().divideCoefficientBy(8), 8);
      BigVec = MIB.buildBitcast(BigTy, BigVec).getReg(0);
      LitVec = MIB.buildBitcast(LitTy, LitVec).getReg(0);
    } else {
      // We can't slide this mask vector up indexed by its i1 elements.
      // This poses a problem when we wish to insert a scalable vector which
      // can't be re-expressed as a larger type. Just choose the slow path and
      // extend to a larger type, then truncate back down.
      LLT ExtBigTy = BigTy.changeElementType(LLT::scalar(8));
      LLT ExtLitTy = LitTy.changeElementType(LLT::scalar(8));
      auto BigZExt = MIB.buildZExt(ExtBigTy, BigVec);
      auto LitZExt = MIB.buildZExt(ExtLitTy, LitVec);
      auto Insert = MIB.buildInsertSubvector(ExtBigTy, BigZExt, LitZExt, Idx);
      auto SplatZero = MIB.buildSplatVector(
          ExtBigTy, MIB.buildConstant(ExtBigTy.getElementType(), 0));
      MIB.buildICmp(CmpInst::Predicate::ICMP_NE, Dst, Insert, SplatZero);
      MI.eraseFromParent();
      return;
    }
  }

  const RISCVRegisterInfo *TRI = STI.getRegisterInfo();
  MVT LitTyMVT = getMVTForLLT(LitTy);
  unsigned SubRegIdx, RemIdx;
  std::tie(SubRegIdx, RemIdx) =
      RISCVTargetLowering::decomposeSubvectorInsertExtractToSubRegs(
          getMVTForLLT(BigTy), LitTyMVT, Idx, TRI);

  RISCVII::VLMUL SubVecLMUL = RISCVTargetLowering::getLMUL(getMVTForLLT(LitTy));
  bool IsSubVecPartReg = !RISCVVType::decodeVLMUL(SubVecLMUL).second;

  // If the Idx has been completely eliminated and this subvector's size is a
  // vector register or a multiple thereof, or the surrounding elements are
  // undef, then this is a subvector insert which naturally aligns to a vector
  // register. These can easily be handled using subregister manipulation.
  if (RemIdx == 0 && (!IsSubVecPartReg || MRI.getVRegDef(Src1)->getOpcode() ==
                                              TargetOpcode::G_IMPLICIT_DEF))
    return;

  // If the subvector is smaller than a vector register, then the insertion
  // must preserve the undisturbed elements of the register. We do this by
  // lowering to an EXTRACT_SUBVECTOR grabbing the nearest LMUL=1 vector type
  // (which resolves to a subregister copy), performing a VSLIDEUP to place the
  // subvector within the vector register, and an INSERT_SUBVECTOR of that
  // LMUL=1 type back into the larger vector (resolving to another subregister
  // operation). See below for how our VSLIDEUP works. We go via a LMUL=1 type
  // to avoid allocating a large register group to hold our subvector.

  // VSLIDEUP works by leaving elements 0<i<OFFSET undisturbed, elements
  // OFFSET<=i<VL set to the "subvector" and vl<=i<VLMAX set to the tail policy
  // (in our case undisturbed). This means we can set up a subvector insertion
  // where OFFSET is the insertion offset, and the VL is the OFFSET plus the
  // size of the subvector.
  const LLT XLenTy(STI.getXLenVT());
  LLT InterLitTy = BigTy;
  Register AlignedExtract = Src1;
  unsigned AlignedIdx = Idx - RemIdx;
  if (TypeSize::isKnownGT(BigTy.getSizeInBits(),
                          getLMUL1Ty(BigTy).getSizeInBits())) {
    InterLitTy = getLMUL1Ty(BigTy);
    // Extract a subvector equal to the nearest full vector register type. This
    // should resolve to a G_EXTRACT on a subreg.
    AlignedExtract =
        MIB.buildExtractSubvector(InterLitTy, BigVec, AlignedIdx).getReg(0);
  }

  auto Insert = MIB.buildInsertSubvector(InterLitTy, MIB.buildUndef(InterLitTy),
                                         LitVec, 0);

  auto [Mask, _] = buildDefaultVLOps(BigTy, MIB, MRI);
  auto VL = MIB.buildVScale(XLenTy, LitTy.getElementCount().getKnownMinValue());

  // Use tail agnostic policy if we're inserting over InterLitTy's tail.
  ElementCount EndIndex =
      ElementCount::getScalable(RemIdx) + LitTy.getElementCount();
  uint64_t Policy = RISCVII::TAIL_UNDISTURBED_MASK_UNDISTURBED;
  if (EndIndex == InterLitTy.getElementCount())
    Policy = RISCVII::TAIL_AGNOSTIC;

  // If we're inserting into the lowest elements, use a tail undisturbed
  // vmv.v.v.
  MachineInstrBuilder Inserted;
  if (RemIdx == 0) {
    Inserted = MIB.buildInstr(RISCV::G_VMV_V_V_VL, {InterLitTy},
                              {AlignedExtract, Insert, VL});
  } else {
    auto SlideupAmt = MIB.buildVScale(XLenTy, RemIdx);
    // Construct the vector length corresponding to RemIdx + length(LitTy).
    VL = MIB.buildAdd(XLenTy, SlideupAmt, VL);
    Inserted =
        MIB.buildInstr(RISCV::G_VSLIDEUP_VL, {InterLitTy},
                       {AlignedExtract, LitVec, SlideupAmt, Mask, VL, Policy});
  }

  // If required, insert this subvector back into the correct vector register.
  // This should resolve to an INSERT_SUBREG instruction.
  if (TypeSize::isKnownGT(BigTy.getSizeInBits(), InterLitTy.getSizeInBits()))
    Inserted = MIB.buildInsertSubvector(BigTy, BigVec, LitVec, AlignedIdx);

  // We might have bitcast from a mask type: cast back to the original type if
  // required.
  MIB.buildBitcast(Dst, Inserted);

  MI.eraseFromParent();
  return;
}

class RISCVPostLegalizerLoweringImpl : public Combiner {
protected:
  // TODO: Make CombinerHelper methods const.
  mutable CombinerHelper Helper;
  const RISCVPostLegalizerLoweringImplRuleConfig &RuleConfig;
  const RISCVSubtarget &STI;

public:
  RISCVPostLegalizerLoweringImpl(
      MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
      GISelCSEInfo *CSEInfo,
      const RISCVPostLegalizerLoweringImplRuleConfig &RuleConfig,
      const RISCVSubtarget &STI);

  static const char *getName() { return "RISCVPreLegalizerCombiner"; }

  bool tryCombineAll(MachineInstr &I) const override;

private:
#define GET_GICOMBINER_CLASS_MEMBERS
#include "RISCVGenPostLegalizeGILowering.inc"
#undef GET_GICOMBINER_CLASS_MEMBERS
};

#define GET_GICOMBINER_IMPL
#include "RISCVGenPostLegalizeGILowering.inc"
#undef GET_GICOMBINER_IMPL

RISCVPostLegalizerLoweringImpl::RISCVPostLegalizerLoweringImpl(
    MachineFunction &MF, CombinerInfo &CInfo, const TargetPassConfig *TPC,
    GISelCSEInfo *CSEInfo,
    const RISCVPostLegalizerLoweringImplRuleConfig &RuleConfig,
    const RISCVSubtarget &STI)
    : Combiner(MF, CInfo, TPC, /*KB*/ nullptr, CSEInfo),
      Helper(Observer, B, /*IsPreLegalize*/ true), RuleConfig(RuleConfig),
      STI(STI),
#define GET_GICOMBINER_CONSTRUCTOR_INITS
#include "RISCVGenPostLegalizeGILowering.inc"
#undef GET_GICOMBINER_CONSTRUCTOR_INITS
{
}

class RISCVPostLegalizerLowering : public MachineFunctionPass {
public:
  static char ID;

  RISCVPostLegalizerLowering();

  StringRef getPassName() const override {
    return "RISCVPostLegalizerLowering";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  RISCVPostLegalizerLoweringImplRuleConfig RuleConfig;
};
} // end anonymous namespace

void RISCVPostLegalizerLowering::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.setPreservesCFG();
  getSelectionDAGFallbackAnalysisUsage(AU);
  MachineFunctionPass::getAnalysisUsage(AU);
}

RISCVPostLegalizerLowering::RISCVPostLegalizerLowering()
    : MachineFunctionPass(ID) {
  if (!RuleConfig.parseCommandLineOption())
    report_fatal_error("Invalid rule identifier");
}

bool RISCVPostLegalizerLowering::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getProperties().hasProperty(
          MachineFunctionProperties::Property::FailedISel))
    return false;
  assert(MF.getProperties().hasProperty(
             MachineFunctionProperties::Property::Legalized) &&
         "Expected a legalized function?");
  auto *TPC = &getAnalysis<TargetPassConfig>();
  const Function &F = MF.getFunction();

  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  CombinerInfo CInfo(/*AllowIllegalOps*/ true, /*ShouldLegalizeIllegal*/ false,
                     /*LegalizerInfo*/ nullptr, /*OptEnabled=*/true,
                     F.hasOptSize(), F.hasMinSize());
  // Disable fixed-point iteration to reduce compile-time
  CInfo.MaxIterations = 1;
  CInfo.ObserverLvl = CombinerInfo::ObserverLevel::SinglePass;
  // PostLegalizerCombiner performs DCE, so a full DCE pass is unnecessary.
  CInfo.EnableFullDCE = false;
  RISCVPostLegalizerLoweringImpl Impl(MF, CInfo, TPC, /*CSEInfo*/ nullptr,
                                      RuleConfig, ST);
  return Impl.combineMachineInstrs();
}

char RISCVPostLegalizerLowering::ID = 0;
INITIALIZE_PASS_BEGIN(RISCVPostLegalizerLowering, DEBUG_TYPE,
                      "Lower RISC-V MachineInstrs after legalization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(RISCVPostLegalizerLowering, DEBUG_TYPE,
                    "Lower RISC-V MachineInstrs after legalization", false,
                    false)

namespace llvm {
FunctionPass *createRISCVPostLegalizerLowering() {
  return new RISCVPostLegalizerLowering();
}
} // end namespace llvm
