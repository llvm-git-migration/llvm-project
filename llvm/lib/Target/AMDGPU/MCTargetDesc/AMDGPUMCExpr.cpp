//===- AMDGPUMCExpr.cpp - AMDGPU specific MC expression classes -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AMDGPUMCExpr.h"
#include "GCNSubtarget.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/KnownBits.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>

using namespace llvm;
using namespace llvm::AMDGPU;

AMDGPUVariadicMCExpr::AMDGPUVariadicMCExpr(VariadicKind Kind,
                                           ArrayRef<const MCExpr *> Args,
                                           MCContext &Ctx)
    : Kind(Kind), Ctx(Ctx) {
  assert(Args.size() >= 1 && "Needs a minimum of one expression.");
  assert(Kind != AGVK_None &&
         "Cannot construct AMDGPUVariadicMCExpr of kind none.");

  // Allocating the variadic arguments through the same allocation mechanism
  // that the object itself is allocated with so they end up in the same memory.
  //
  // Will result in an asan failure if allocated on the heap through standard
  // allocation (e.g., through SmallVector's grow).
  RawArgs = static_cast<const MCExpr **>(
      Ctx.allocate(sizeof(const MCExpr *) * Args.size()));
  std::uninitialized_copy(Args.begin(), Args.end(), RawArgs);
  this->Args = ArrayRef<const MCExpr *>(RawArgs, Args.size());
}

AMDGPUVariadicMCExpr::~AMDGPUVariadicMCExpr() { Ctx.deallocate(RawArgs); }

const AMDGPUVariadicMCExpr *
AMDGPUVariadicMCExpr::create(VariadicKind Kind, ArrayRef<const MCExpr *> Args,
                             MCContext &Ctx) {
  return new (Ctx) AMDGPUVariadicMCExpr(Kind, Args, Ctx);
}

const MCExpr *AMDGPUVariadicMCExpr::getSubExpr(size_t Index) const {
  assert(Index < Args.size() &&
         "Indexing out of bounds AMDGPUVariadicMCExpr sub-expr");
  return Args[Index];
}

void AMDGPUVariadicMCExpr::printImpl(raw_ostream &OS,
                                     const MCAsmInfo *MAI) const {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown AMDGPUVariadicMCExpr kind.");
  case AGVK_Or:
    OS << "or(";
    break;
  case AGVK_Max:
    OS << "max(";
    break;
  case AGVK_ExtraSGPRs:
    OS << "extrasgprs(";
    break;
  case AGVK_TotalNumVGPRs:
    OS << "totalnumvgprs(";
    break;
  case AGVK_AlignTo:
    OS << "alignto(";
    break;
  case AGVK_Occupancy:
    OS << "occupancy(";
    break;
  }
  for (auto It = Args.begin(); It != Args.end(); ++It) {
    (*It)->print(OS, MAI, /*InParens=*/false);
    if ((It + 1) != Args.end())
      OS << ", ";
  }
  OS << ')';
}

static int64_t op(AMDGPUVariadicMCExpr::VariadicKind Kind, int64_t Arg1,
                  int64_t Arg2) {
  switch (Kind) {
  default:
    llvm_unreachable("Unknown AMDGPUVariadicMCExpr kind.");
  case AMDGPUVariadicMCExpr::AGVK_Max:
    return std::max(Arg1, Arg2);
  case AMDGPUVariadicMCExpr::AGVK_Or:
    return Arg1 | Arg2;
  }
}

bool AMDGPUVariadicMCExpr::evaluateExtraSGPRs(MCValue &Res,
                                              const MCAsmLayout *Layout,
                                              const MCFixup *Fixup) const {
  auto TryGetMCExprValue = [&](const MCExpr *Arg, uint64_t &ConstantValue) {
    MCValue MCVal;
    if (!Arg->evaluateAsRelocatable(MCVal, Layout, Fixup) ||
        !MCVal.isAbsolute())
      return false;

    ConstantValue = MCVal.getConstant();
    return true;
  };

  assert(Args.size() == 3 &&
         "AMDGPUVariadic Argument count incorrect for ExtraSGPRs");
  const MCSubtargetInfo *STI = Ctx.getSubtargetInfo();
  uint64_t VCCUsed = 0, FlatScrUsed = 0, XNACKUsed = 0;

  bool Success = TryGetMCExprValue(Args[2], XNACKUsed);

  assert(Success && "Arguments 3 for ExtraSGPRs should be a known constant");
  if (!Success || !TryGetMCExprValue(Args[0], VCCUsed) ||
      !TryGetMCExprValue(Args[1], FlatScrUsed))
    return false;

  uint64_t ExtraSGPRs = IsaInfo::getNumExtraSGPRs(
      STI, (bool)VCCUsed, (bool)FlatScrUsed, (bool)XNACKUsed);
  Res = MCValue::get(ExtraSGPRs);
  return true;
}

bool AMDGPUVariadicMCExpr::evaluateTotalNumVGPR(MCValue &Res,
                                                const MCAsmLayout *Layout,
                                                const MCFixup *Fixup) const {
  auto TryGetMCExprValue = [&](const MCExpr *Arg, uint64_t &ConstantValue) {
    MCValue MCVal;
    if (!Arg->evaluateAsRelocatable(MCVal, Layout, Fixup) ||
        !MCVal.isAbsolute())
      return false;

    ConstantValue = MCVal.getConstant();
    return true;
  };
  assert(Args.size() == 2 &&
         "AMDGPUVariadic Argument count incorrect for TotalNumVGPRs");
  const MCSubtargetInfo *STI = Ctx.getSubtargetInfo();
  uint64_t NumAGPR = 0, NumVGPR = 0;

  bool Has90AInsts = AMDGPU::isGFX90A(*STI);

  if (!TryGetMCExprValue(Args[0], NumAGPR) ||
      !TryGetMCExprValue(Args[1], NumVGPR))
    return false;

  uint64_t TotalNum = Has90AInsts && NumAGPR ? alignTo(NumVGPR, 4) + NumAGPR
                                             : std::max(NumVGPR, NumAGPR);
  Res = MCValue::get(TotalNum);
  return true;
}

bool AMDGPUVariadicMCExpr::evaluateAlignTo(MCValue &Res,
                                           const MCAsmLayout *Layout,
                                           const MCFixup *Fixup) const {
  auto TryGetMCExprValue = [&](const MCExpr *Arg, uint64_t &ConstantValue) {
    MCValue MCVal;
    if (!Arg->evaluateAsRelocatable(MCVal, Layout, Fixup) ||
        !MCVal.isAbsolute())
      return false;

    ConstantValue = MCVal.getConstant();
    return true;
  };

  assert(Args.size() == 2 &&
         "AMDGPUVariadic Argument count incorrect for AlignTo");
  uint64_t Value = 0, Align = 0;
  if (!TryGetMCExprValue(Args[0], Value) || !TryGetMCExprValue(Args[1], Align))
    return false;

  Res = MCValue::get(alignTo(Value, Align));
  return true;
}

bool AMDGPUVariadicMCExpr::evaluateOccupancy(MCValue &Res,
                                             const MCAsmLayout *Layout,
                                             const MCFixup *Fixup) const {
  auto TryGetMCExprValue = [&](const MCExpr *Arg, uint64_t &ConstantValue) {
    MCValue MCVal;
    if (!Arg->evaluateAsRelocatable(MCVal, Layout, Fixup) ||
        !MCVal.isAbsolute())
      return false;

    ConstantValue = MCVal.getConstant();
    return true;
  };
  assert(Args.size() == 7 &&
         "AMDGPUVariadic Argument count incorrect for Occupancy");
  uint64_t InitOccupancy, MaxWaves, Granule, TargetTotalNumVGPRs, Generation,
      NumSGPRs, NumVGPRs;

  bool Success = true;
  Success &= TryGetMCExprValue(Args[0], MaxWaves);
  Success &= TryGetMCExprValue(Args[1], Granule);
  Success &= TryGetMCExprValue(Args[2], TargetTotalNumVGPRs);
  Success &= TryGetMCExprValue(Args[3], Generation);
  Success &= TryGetMCExprValue(Args[4], InitOccupancy);

  assert(Success && "Arguments 1 to 5 for Occupancy should be known constants");

  if (!Success || !TryGetMCExprValue(Args[5], NumSGPRs) ||
      !TryGetMCExprValue(Args[6], NumVGPRs))
    return false;

  unsigned Occupancy = InitOccupancy;
  if (NumSGPRs)
    Occupancy = std::min(
        Occupancy, IsaInfo::getOccupancyWithNumSGPRs(
                       NumSGPRs, MaxWaves,
                       static_cast<AMDGPUSubtarget::Generation>(Generation)));
  if (NumVGPRs)
    Occupancy = std::min(Occupancy,
                         IsaInfo::getNumWavesPerEUWithNumVGPRs(
                             NumVGPRs, Granule, MaxWaves, TargetTotalNumVGPRs));

  Res = MCValue::get(Occupancy);
  return true;
}

bool AMDGPUVariadicMCExpr::evaluateAsRelocatableImpl(
    MCValue &Res, const MCAsmLayout *Layout, const MCFixup *Fixup) const {
  std::optional<int64_t> Total;

  switch (Kind) {
  default:
    break;
  case AGVK_ExtraSGPRs:
    return evaluateExtraSGPRs(Res, Layout, Fixup);
  case AGVK_AlignTo:
    return evaluateAlignTo(Res, Layout, Fixup);
  case AGVK_TotalNumVGPRs:
    return evaluateTotalNumVGPR(Res, Layout, Fixup);
  case AGVK_Occupancy:
    return evaluateOccupancy(Res, Layout, Fixup);
  }

  for (const MCExpr *Arg : Args) {
    MCValue ArgRes;
    if (!Arg->evaluateAsRelocatable(ArgRes, Layout, Fixup) ||
        !ArgRes.isAbsolute())
      return false;

    if (!Total.has_value())
      Total = ArgRes.getConstant();
    Total = op(Kind, *Total, ArgRes.getConstant());
  }

  Res = MCValue::get(*Total);
  return true;
}

void AMDGPUVariadicMCExpr::visitUsedExpr(MCStreamer &Streamer) const {
  for (const MCExpr *Arg : Args)
    Streamer.visitUsedExpr(*Arg);
}

MCFragment *AMDGPUVariadicMCExpr::findAssociatedFragment() const {
  for (const MCExpr *Arg : Args) {
    if (Arg->findAssociatedFragment())
      return Arg->findAssociatedFragment();
  }
  return nullptr;
}

/// Allow delayed MCExpr resolve of ExtraSGPRs (in case VCCUsed or FlatScrUsed
/// are unresolvable but needed for further MCExprs). Derived from
/// implementation of IsaInfo::getNumExtraSGPRs in AMDGPUBaseInfo.cpp.
///
const AMDGPUVariadicMCExpr *
AMDGPUVariadicMCExpr::createExtraSGPRs(const MCExpr *VCCUsed,
                                       const MCExpr *FlatScrUsed,
                                       bool XNACKUsed, MCContext &Ctx) {

  return create(AGVK_ExtraSGPRs,
                {VCCUsed, FlatScrUsed, MCConstantExpr::create(XNACKUsed, Ctx)},
                Ctx);
}

const AMDGPUVariadicMCExpr *AMDGPUVariadicMCExpr::createTotalNumVGPR(
    const MCExpr *NumAGPR, const MCExpr *NumVGPR, MCContext &Ctx) {
  return create(AGVK_TotalNumVGPRs, {NumAGPR, NumVGPR}, Ctx);
}

/// Mimics GCNSubtarget::computeOccupancy for MCExpr.
///
/// Remove dependency on GCNSubtarget and depend only only the necessary values
/// for said occupancy computation. Should match computeOccupancy implementation
/// without passing \p STM on.
const AMDGPUVariadicMCExpr *
AMDGPUVariadicMCExpr::createOccupancy(unsigned InitOcc, const MCExpr *NumSGPRs,
                                      const MCExpr *NumVGPRs,
                                      const GCNSubtarget &STM, MCContext &Ctx) {
  unsigned MaxWaves = IsaInfo::getMaxWavesPerEU(&STM);
  unsigned Granule = IsaInfo::getVGPRAllocGranule(&STM);
  unsigned TargetTotalNumVGPRs = IsaInfo::getTotalNumVGPRs(&STM);
  unsigned Generation = STM.getGeneration();

  auto CreateExpr = [&Ctx](unsigned Value) {
    return MCConstantExpr::create(Value, Ctx);
  };

  return create(AGVK_Occupancy,
                {CreateExpr(MaxWaves), CreateExpr(Granule),
                 CreateExpr(TargetTotalNumVGPRs), CreateExpr(Generation),
                 CreateExpr(InitOcc), NumSGPRs, NumVGPRs},
                Ctx);
}

static KnownBits AMDGPUMCExprKnownBits(const MCExpr *Expr, raw_ostream &OS,
                                       const MCAsmInfo *MAI, unsigned depth) {

  if (depth == 0)
    return KnownBits(/*BitWidth=*/64);

  depth--;

  switch (Expr->getKind()) {
  case MCExpr::ExprKind::Binary: {
    const MCBinaryExpr *BExpr = cast<MCBinaryExpr>(Expr);
    const MCExpr *LHS = BExpr->getLHS();
    const MCExpr *RHS = BExpr->getRHS();

    KnownBits LHSKnown = AMDGPUMCExprKnownBits(LHS, OS, MAI, depth);
    KnownBits RHSKnown = AMDGPUMCExprKnownBits(RHS, OS, MAI, depth);

    switch (BExpr->getOpcode()) {
    default:
      return KnownBits(/*BitWidth=*/64);
    case MCBinaryExpr::Opcode::Add:
      return KnownBits::computeForAddSub(/*Add=*/true, /*NSW=*/false,
                                         /*NUW=*/false, LHSKnown, RHSKnown);
    case MCBinaryExpr::Opcode::And:
      return LHSKnown & RHSKnown;
    case MCBinaryExpr::Opcode::Div:
      return KnownBits::sdiv(LHSKnown, RHSKnown);
    case MCBinaryExpr::Opcode::Mod:
      return KnownBits::srem(LHSKnown, RHSKnown);
    case MCBinaryExpr::Opcode::Mul:
      return KnownBits::mul(LHSKnown, RHSKnown);
    case MCBinaryExpr::Opcode::Or:
      return LHSKnown | RHSKnown;
    case MCBinaryExpr::Opcode::Shl:
      return KnownBits::shl(LHSKnown, RHSKnown);
    case MCBinaryExpr::Opcode::AShr:
      return KnownBits::ashr(LHSKnown, RHSKnown);
    case MCBinaryExpr::Opcode::LShr:
      return KnownBits::lshr(LHSKnown, RHSKnown);
    case MCBinaryExpr::Opcode::Sub:
      return KnownBits::computeForAddSub(/*Add=*/false, /*NSW=*/false,
                                         /*NUW=*/false, LHSKnown, RHSKnown);
    case MCBinaryExpr::Opcode::Xor:
      return LHSKnown ^ RHSKnown;
    }
  }
  case MCExpr::ExprKind::Constant: {
    const MCConstantExpr *CE = cast<MCConstantExpr>(Expr);
    APInt APValue(/*BitWidth=*/64, CE->getValue(), /*isSigned=*/true);
    return KnownBits::makeConstant(APValue);
  }
  case MCExpr::ExprKind::SymbolRef: {
    const MCSymbolRefExpr *RExpr = cast<MCSymbolRefExpr>(Expr);
    const MCSymbol &Sym = RExpr->getSymbol();
    if (!Sym.isVariable())
      return KnownBits(/*BitWidth=*/64);

    // Variable value retrieval is not for actual use but only for knownbits
    // analysis.
    return AMDGPUMCExprKnownBits(Sym.getVariableValue(/*SetUsed=*/false), OS,
                                 MAI, depth);
  }
  case MCExpr::ExprKind::Unary: {
    const MCUnaryExpr *UExpr = cast<MCUnaryExpr>(Expr);
    KnownBits KB = AMDGPUMCExprKnownBits(UExpr->getSubExpr(), OS, MAI, depth);

    switch (UExpr->getOpcode()) {
    default:
      return KnownBits(/*BitWidth=*/64);
    case MCUnaryExpr::Opcode::Minus: {
      KB.makeNegative();
      return KB;
    }
    case MCUnaryExpr::Opcode::Not: {
      KnownBits AllOnes(/*BitWidth=*/64);
      AllOnes.setAllOnes();
      return KB ^ AllOnes;
    }
    case MCUnaryExpr::Opcode::Plus: {
      KB.makeNonNegative();
      return KB;
    }
    }
  }
  case MCExpr::ExprKind::Target: {
    const AMDGPUVariadicMCExpr *AGVK = cast<AMDGPUVariadicMCExpr>(Expr);

    switch (AGVK->getKind()) {
    default:
      return KnownBits(/*BitWidth=*/64);
    case AMDGPUVariadicMCExpr::VariadicKind::AGVK_Or: {
      KnownBits KB = AMDGPUMCExprKnownBits(AGVK->getSubExpr(0), OS, MAI, depth);
      for (const MCExpr *Arg : AGVK->getArgs()) {
        KB |= AMDGPUMCExprKnownBits(Arg, OS, MAI, depth);
      }
      return KB;
    }
    case AMDGPUVariadicMCExpr::VariadicKind::AGVK_Max: {
      KnownBits KB = AMDGPUMCExprKnownBits(AGVK->getSubExpr(0), OS, MAI, depth);
      for (const MCExpr *Arg : AGVK->getArgs()) {
        KB = KnownBits::umax(KB, AMDGPUMCExprKnownBits(Arg, OS, MAI, depth));
      }
      return KB;
    }
    case AMDGPUVariadicMCExpr::VariadicKind::AGVK_ExtraSGPRs:
    case AMDGPUVariadicMCExpr::VariadicKind::AGVK_TotalNumVGPRs:
    case AMDGPUVariadicMCExpr::VariadicKind::AGVK_AlignTo:
    case AMDGPUVariadicMCExpr::VariadicKind::AGVK_Occupancy: {
      int64_t Val;
      if (AGVK->evaluateAsAbsolute(Val)) {
        APInt APValue(/*BitWidth=*/64, Val, /*isSigned=*/false);
        return KnownBits::makeConstant(APValue);
      } else {
        return KnownBits(/*BitWidth=*/64);
      }
    }
    }
  }
  }
  return KnownBits(/*BitWidth=*/64);
}

void llvm::AMDGPUMCExprPrint(const MCExpr *Expr, raw_ostream &OS,
                             const MCAsmInfo *MAI) {
  int64_t Val;
  if (Expr->evaluateAsAbsolute(Val)) {
    OS << Val;
    return;
  }

  KnownBits KB = AMDGPUMCExprKnownBits(Expr, OS, MAI, /*depth=*/16);
  if (KB.isConstant()) {
    OS << KB.getConstant();
    return;
  }

  Expr->print(OS, MAI);
}
