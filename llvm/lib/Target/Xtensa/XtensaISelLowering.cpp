//===- XtensaISelLowering.cpp - Xtensa DAG Lowering Implementation --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that Xtensa uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#include "XtensaISelLowering.h"
#include "XtensaConstantPoolValue.h"
#include "XtensaInstrInfo.h"
#include "XtensaMachineFunctionInfo.h"
#include "XtensaSubtarget.h"
#include "XtensaTargetMachine.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <deque>

using namespace llvm;

#define DEBUG_TYPE "xtensa-lower"

// Return true if we must use long (in fact, indirect) function call.
// It's simplified version, production implimentation must
// resolve a functions in ROM (usually glibc functions)
static bool isLongCall(const char *str) {
  // Currently always use long calls
  return true;
}

XtensaTargetLowering::XtensaTargetLowering(const TargetMachine &TM,
                                           const XtensaSubtarget &STI)
    : TargetLowering(TM), Subtarget(STI) {
  MVT PtrVT = MVT::i32;
  // Set up the register classes.
  addRegisterClass(MVT::i32, &Xtensa::ARRegClass);

  // Set up special registers.
  setStackPointerRegisterToSaveRestore(Xtensa::SP);

  setSchedulingPreference(Sched::RegPressure);

  setMinFunctionAlignment(Align(4));

  setOperationAction(ISD::Constant, MVT::i32, Custom);
  setOperationAction(ISD::Constant, MVT::i64, Expand);

  setBooleanContents(ZeroOrOneBooleanContent);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8, Expand);
  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);

  setOperationAction(ISD::BITCAST, MVT::i32, Expand);
  setOperationAction(ISD::BITCAST, MVT::f32, Expand);
  setOperationAction(ISD::UINT_TO_FP, MVT::i32, Expand);
  setOperationAction(ISD::SINT_TO_FP, MVT::i32, Expand);
  setOperationAction(ISD::FP_TO_UINT, MVT::i32, Expand);
  setOperationAction(ISD::FP_TO_SINT, MVT::i32, Expand);

  // No sign extend instructions for i1 and sign extend load i8
  for (MVT VT : MVT::integer_valuetypes()) {
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::ZEXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::EXTLOAD, VT, MVT::i1, Promote);
    setLoadExtAction(ISD::SEXTLOAD, VT, MVT::i8, Expand);
  }

  setOperationAction(ISD::ConstantPool, PtrVT, Custom);
  setOperationAction(ISD::GlobalAddress, PtrVT, Custom);
  setOperationAction(ISD::BlockAddress, PtrVT, Custom);
  setOperationAction(ISD::JumpTable, PtrVT, Custom);

  // Expand jump table branches as address arithmetic followed by an
  // indirect jump.
  setOperationAction(ISD::BR_JT, MVT::Other, Custom);

  setOperationAction(ISD::BR_CC, MVT::i32, Legal);
  setOperationAction(ISD::BR_CC, MVT::i64, Expand);
  setOperationAction(ISD::BR_CC, MVT::f32, Expand);

  setOperationAction(ISD::SELECT, MVT::i32, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);
  setOperationAction(ISD::SETCC, MVT::i32, Expand);

  setCondCodeAction(ISD::SETGT, MVT::i32, Expand);
  setCondCodeAction(ISD::SETLE, MVT::i32, Expand);
  setCondCodeAction(ISD::SETUGT, MVT::i32, Expand);
  setCondCodeAction(ISD::SETULE, MVT::i32, Expand);

  setOperationAction(ISD::MUL, MVT::i32, Expand);
  setOperationAction(ISD::MULHU, MVT::i32, Expand);
  setOperationAction(ISD::MULHS, MVT::i32, Expand);
  setOperationAction(ISD::SMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i32, Expand);

  setOperationAction(ISD::SDIV, MVT::i32, Expand);
  setOperationAction(ISD::UDIV, MVT::i32, Expand);
  setOperationAction(ISD::SREM, MVT::i32, Expand);
  setOperationAction(ISD::UREM, MVT::i32, Expand);
  setOperationAction(ISD::SDIVREM, MVT::i32, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i32, Expand);

  setOperationAction(ISD::SHL_PARTS, MVT::i32, Custom);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Custom);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Custom);

  setOperationAction(ISD::BSWAP, MVT::i32, Expand);
  setOperationAction(ISD::ROTL, MVT::i32, Expand);
  setOperationAction(ISD::ROTR, MVT::i32, Expand);
  setOperationAction(ISD::CTPOP, MVT::i32, Custom);
  setOperationAction(ISD::CTTZ, MVT::i32, Expand);
  setOperationAction(ISD::CTLZ, MVT::i32, Expand);
  setOperationAction(ISD::CTTZ_ZERO_UNDEF, MVT::i32, Expand);
  setOperationAction(ISD::CTLZ_ZERO_UNDEF, MVT::i32, Expand);

  // Implement custom stack allocations
  setOperationAction(ISD::DYNAMIC_STACKALLOC, PtrVT, Custom);
  // Implement custom stack save and restore
  setOperationAction(ISD::STACKSAVE, MVT::Other, Custom);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Custom);

  // VASTART and VACOPY need to deal with the Xtensa-specific varargs
  // structure, but VAEND is a no-op.
  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  // we use special va_list structure so we have to customize this
  setOperationAction(ISD::VAARG, MVT::Other, Custom);
  setOperationAction(ISD::VACOPY, MVT::Other, Custom);
  setOperationAction(ISD::VAEND, MVT::Other, Expand);

  // Compute derived properties from the register classes
  computeRegisterProperties(STI.getRegisterInfo());
}

bool XtensaTargetLowering::isOffsetFoldingLegal(
    const GlobalAddressSDNode *GA) const {
  // The Xtensa target isn't yet aware of offsets.
  return false;
}

//===----------------------------------------------------------------------===//
// Inline asm support
//===----------------------------------------------------------------------===//
TargetLowering::ConstraintType
XtensaTargetLowering::getConstraintType(StringRef Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'r':
      return C_RegisterClass;
    default:
      break;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

TargetLowering::ConstraintWeight
XtensaTargetLowering::getSingleConstraintMatchWeight(
    AsmOperandInfo &Info, const char *Constraint) const {
  ConstraintWeight Weight = CW_Invalid;
  Value *CallOperandVal = Info.CallOperandVal;
  // If we don't have a value, we can't do a match,
  // but allow it at the lowest weight.
  if (!CallOperandVal)
    return CW_Default;

  Type *Ty = CallOperandVal->getType();

  // Look at the constraint type.
  switch (*Constraint) {
  default:
    Weight = TargetLowering::getSingleConstraintMatchWeight(Info, Constraint);
    break;
  case 'r':
    if (Ty->isIntegerTy())
      Weight = CW_Register;
    break;
  }
  return Weight;
}

std::pair<unsigned, const TargetRegisterClass *>
XtensaTargetLowering::getRegForInlineAsmConstraint(
    const TargetRegisterInfo *TRI, StringRef Constraint, MVT VT) const {
  if (Constraint.size() == 1) {
    // GCC Constraint Letters
    switch (Constraint[0]) {
    default:
      break;
    case 'r': // General-purpose register
      return std::make_pair(0U, &Xtensa::ARRegClass);
    }
  }
  return TargetLowering::getRegForInlineAsmConstraint(TRI, Constraint, VT);
}

void XtensaTargetLowering::LowerAsmOperandForConstraint(
    SDValue Op, StringRef Constraint, std::vector<SDValue> &Ops,
    SelectionDAG &DAG) const {
  SDLoc DL(Op);

  // Only support length 1 constraints for now.
  if (Constraint.size() > 1)
    return;

  TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

unsigned XtensaTargetLowering::getVaListSizeInBits(const DataLayout &DL) const {
  // 2 * sizeof(int*) + sizeof(int)
  return 3 * 4;
}

//===----------------------------------------------------------------------===//
// Calling conventions
//===----------------------------------------------------------------------===//

#include "XtensaGenCallingConv.inc"

static bool CC_Xtensa_Custom(unsigned ValNo, MVT ValVT, MVT LocVT,
                             CCValAssign::LocInfo LocInfo,
                             ISD::ArgFlagsTy ArgFlags, CCState &State) {
  static const MCPhysReg IntRegs[] = {Xtensa::A2, Xtensa::A3, Xtensa::A4,
                                      Xtensa::A5, Xtensa::A6, Xtensa::A7};

  if (ArgFlags.isByVal()) {
    Align ByValAlign = ArgFlags.getNonZeroByValAlign();
    unsigned ByValSize = ArgFlags.getByValSize();
    if (ByValSize < 4) {
      ByValSize = 4;
    }
    if (ByValAlign < Align(4)) {
      ByValAlign = Align(4);
    }
    unsigned Offset = State.AllocateStack(ByValSize, ByValAlign);
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset, LocVT, LocInfo));
    // Mark all unused registers as allocated to avoid misuse
    // of such registers.
    while (State.AllocateReg(IntRegs))
      ;
    return false;
  }

  // Promote i8 and i16
  if (LocVT == MVT::i8 || LocVT == MVT::i16) {
    LocVT = MVT::i32;
    if (ArgFlags.isSExt())
      LocInfo = CCValAssign::SExt;
    else if (ArgFlags.isZExt())
      LocInfo = CCValAssign::ZExt;
    else
      LocInfo = CCValAssign::AExt;
  }

  unsigned Register;

  Align OrigAlign = ArgFlags.getNonZeroOrigAlign();
  bool needs64BitAlign = (ValVT == MVT::i32 && OrigAlign == Align(8));
  bool needs128BitAlign = (ValVT == MVT::i32 && OrigAlign == Align(16));

  if (ValVT == MVT::i32) {
    Register = State.AllocateReg(IntRegs);
    // If this is the first part of an i64 arg,
    // the allocated register must be either A2, A4 or A6.
    if (needs64BitAlign && (Register == Xtensa::A3 || Register == Xtensa::A5 ||
                            Register == Xtensa::A7))
      Register = State.AllocateReg(IntRegs);
    // arguments with 16byte alignment must be passed in the first register or
    // passed via stack
    if (needs128BitAlign && (Register != Xtensa::A2))
      while ((Register = State.AllocateReg(IntRegs)))
        ;
    LocVT = MVT::i32;
  } else if (ValVT == MVT::f64) {
    // Allocate int register and shadow next int register.
    Register = State.AllocateReg(IntRegs);
    if (Register == Xtensa::A3 || Register == Xtensa::A5 ||
        Register == Xtensa::A7)
      Register = State.AllocateReg(IntRegs);
    State.AllocateReg(IntRegs);
    LocVT = MVT::i32;
  } else {
    report_fatal_error("Cannot handle this ValVT.");
  }

  if (!Register) {
    unsigned Offset = State.AllocateStack(ValVT.getStoreSize(), OrigAlign);
    State.addLoc(CCValAssign::getMem(ValNo, ValVT, Offset, LocVT, LocInfo));
  } else {
    State.addLoc(CCValAssign::getReg(ValNo, ValVT, Register, LocVT, LocInfo));
  }

  return false;
}

CCAssignFn *XtensaTargetLowering::CCAssignFnForCall(CallingConv::ID CC,
                                                    bool IsVarArg) const {
  return CC_Xtensa_Custom;
}

SDValue XtensaTargetLowering::LowerFormalArguments(
    SDValue Chain, CallingConv::ID CallConv, bool IsVarArg,
    const SmallVectorImpl<ISD::InputArg> &Ins, const SDLoc &DL,
    SelectionDAG &DAG, SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  XtensaMachineFunctionInfo *XtensaFI = MF.getInfo<XtensaMachineFunctionInfo>();
  EVT PtrVT = getPointerTy(MF.getDataLayout());

  XtensaFI->setVarArgsFrameIndex(0);

  // Used with vargs to acumulate store chains.
  std::vector<SDValue> OutChains;

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(), ArgLocs,
                 *DAG.getContext());

  CCInfo.AnalyzeFormalArguments(Ins, CCAssignFnForCall(CallConv, IsVarArg));

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    // Arguments stored on registers
    if (VA.isRegLoc()) {
      EVT RegVT = VA.getLocVT();
      const TargetRegisterClass *RC;

      if (RegVT == MVT::i32)
        RC = &Xtensa::ARRegClass;
      else
        report_fatal_error("RegVT not supported by FormalArguments Lowering");

      // Transform the arguments stored on
      // physical registers into virtual ones
      unsigned Register = MF.addLiveIn(VA.getLocReg(), RC);
      SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, Register, RegVT);

      // If this is an 8 or 16-bit value, it has been passed promoted
      // to 32 bits.  Insert an assert[sz]ext to capture this, then
      // truncate to the right size.
      if (VA.getLocInfo() != CCValAssign::Full) {
        unsigned Opcode = 0;
        if (VA.getLocInfo() == CCValAssign::SExt)
          Opcode = ISD::AssertSext;
        else if (VA.getLocInfo() == CCValAssign::ZExt)
          Opcode = ISD::AssertZext;
        if (Opcode)
          ArgValue = DAG.getNode(Opcode, DL, RegVT, ArgValue,
                                 DAG.getValueType(VA.getValVT()));
        ArgValue = DAG.getNode((VA.getValVT() == MVT::f32) ? ISD::BITCAST
                                                           : ISD::TRUNCATE,
                               DL, VA.getValVT(), ArgValue);
      }

      InVals.push_back(ArgValue);

    } else {
      assert(VA.isMemLoc());

      EVT ValVT = VA.getValVT();

      // The stack pointer offset is relative to the caller stack frame.
      int FI = MFI.CreateFixedObject(ValVT.getStoreSize(), VA.getLocMemOffset(),
                                     true);

      if (Ins[VA.getValNo()].Flags.isByVal()) {
        // Assume that in this case load operation is created
        SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
        InVals.push_back(FIN);
      } else {
        // Create load nodes to retrieve arguments from the stack
        SDValue FIN =
            DAG.getFrameIndex(FI, getFrameIndexTy(DAG.getDataLayout()));
        InVals.push_back(DAG.getLoad(
            ValVT, DL, Chain, FIN,
            MachinePointerInfo::getFixedStack(DAG.getMachineFunction(), FI)));
      }
    }
  }

  if (IsVarArg) {
    static const MCPhysReg XtensaArgRegs[6] = {
        Xtensa::A2, Xtensa::A3, Xtensa::A4, Xtensa::A5, Xtensa::A6, Xtensa::A7};
    ArrayRef<MCPhysReg> ArgRegs = ArrayRef(XtensaArgRegs);
    unsigned Idx = CCInfo.getFirstUnallocated(ArgRegs);
    const TargetRegisterClass *RC = &Xtensa::ARRegClass;
    MachineFrameInfo &MFI = MF.getFrameInfo();
    MachineRegisterInfo &RegInfo = MF.getRegInfo();
    unsigned RegSize = 4;
    MVT RegTy = MVT::getIntegerVT(RegSize * 8);

    XtensaFI->setVarArgsFirstGPR(Idx + 2); // 2 - number of a2 register

    XtensaFI->setVarArgsStackOffset(MFI.CreateFixedObject(
        PtrVT.getSizeInBits() / 8, CCInfo.getStackSize(), true));

    // Offset of the first variable argument from stack pointer, and size of
    // the vararg save area. For now, the varargs save area is either zero or
    // large enough to hold a0-a7.
    int VaArgOffset, VarArgsSaveSize;

    // If all registers are allocated, then all varargs must be passed on the
    // stack and we don't need to save any argregs.
    if (ArgRegs.size() == Idx) {
      VaArgOffset = CCInfo.getStackSize();
      VarArgsSaveSize = 0;
    } else {
      VarArgsSaveSize = RegSize * (ArgRegs.size() - Idx);
      VaArgOffset = -VarArgsSaveSize;
    }

    // Record the frame index of the first variable argument
    // which is a value necessary to VASTART.
    int FI = MFI.CreateFixedObject(RegSize, VaArgOffset, true);
    XtensaFI->setVarArgsFrameIndex(FI);

    // Copy the integer registers that may have been used for passing varargs
    // to the vararg save area.
    for (unsigned I = Idx; I < ArgRegs.size(); ++I, VaArgOffset += RegSize) {
      const unsigned Reg = RegInfo.createVirtualRegister(RC);
      unsigned FrameReg = Subtarget.getRegisterInfo()->getFrameRegister(MF);

      // Argument passed in FrameReg we save in A8 (in emitPrologue),
      // so load argument from A8
      if (ArgRegs[I] == FrameReg) {
        RegInfo.addLiveIn(Xtensa::A8, Reg);
      } else {
        RegInfo.addLiveIn(ArgRegs[I], Reg);
      }

      SDValue ArgValue = DAG.getCopyFromReg(Chain, DL, Reg, RegTy);
      FI = MFI.CreateFixedObject(RegSize, VaArgOffset, true);
      SDValue PtrOff = DAG.getFrameIndex(FI, getPointerTy(DAG.getDataLayout()));
      SDValue Store = DAG.getStore(Chain, DL, ArgValue, PtrOff,
                                   MachinePointerInfo::getFixedStack(MF, FI));
      cast<StoreSDNode>(Store.getNode())
          ->getMemOperand()
          ->setValue((Value *)nullptr);
      OutChains.push_back(Store);
    }
  }

  // All stores are grouped in one node to allow the matching between
  // the size of Ins and InVals. This only happens when on varg functions
  if (!OutChains.empty()) {
    OutChains.push_back(Chain);
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, OutChains);
  }

  return Chain;
}

SDValue
XtensaTargetLowering::LowerCall(CallLoweringInfo &CLI,
                                SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG = CLI.DAG;
  SDLoc &DL = CLI.DL;
  SmallVector<ISD::OutputArg, 32> &Outs = CLI.Outs;
  SmallVector<SDValue, 32> &OutVals = CLI.OutVals;
  SmallVector<ISD::InputArg, 32> &Ins = CLI.Ins;
  SDValue Chain = CLI.Chain;
  SDValue Callee = CLI.Callee;
  bool &IsTailCall = CLI.IsTailCall;
  CallingConv::ID CallConv = CLI.CallConv;
  bool IsVarArg = CLI.IsVarArg;

  MachineFunction &MF = DAG.getMachineFunction();
  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  const TargetFrameLowering *TFL = Subtarget.getFrameLowering();

  // TODO: Support tail call optimization.
  IsTailCall = false;

  // Analyze the operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, ArgLocs, *DAG.getContext());

  CCAssignFn *CC = CCAssignFnForCall(CallConv, IsVarArg);

  CCInfo.AnalyzeCallOperands(Outs, CC);

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getStackSize();

  Align StackAlignment = TFL->getStackAlign();
  unsigned NextStackOffset = alignTo(NumBytes, StackAlignment);

  Chain = DAG.getCALLSEQ_START(Chain, NextStackOffset, 0, DL);

  // Copy argument values to their designated locations.
  std::deque<std::pair<unsigned, SDValue>> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;
  SDValue StackPtr;
  for (unsigned I = 0, E = ArgLocs.size(); I != E; ++I) {
    CCValAssign &VA = ArgLocs[I];
    SDValue ArgValue = OutVals[I];
    ISD::ArgFlagsTy Flags = Outs[I].Flags;

    if (VA.isRegLoc())
      // Queue up the argument copies and emit them at the end.
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), ArgValue));
    else if (Flags.isByVal()) {
      assert(VA.isMemLoc());
      assert(Flags.getByValSize() &&
             "ByVal args of size 0 should have been ignored by front-end.");
      assert(!IsTailCall &&
             "Do not tail-call optimize if there is a byval argument.");

      if (!StackPtr.getNode())
        StackPtr = DAG.getCopyFromReg(Chain, DL, Xtensa::SP, PtrVT);
      unsigned Offset = VA.getLocMemOffset();
      SDValue Address = DAG.getNode(ISD::ADD, DL, PtrVT, StackPtr,
                                    DAG.getIntPtrConstant(Offset, DL));
      SDValue SizeNode = DAG.getConstant(Flags.getByValSize(), DL, MVT::i32);
      SDValue Memcpy = DAG.getMemcpy(
          Chain, DL, Address, ArgValue, SizeNode, Flags.getNonZeroByValAlign(),
          /*isVolatile=*/false, /*AlwaysInline=*/false,
          /*CI=*/nullptr, std::nullopt, MachinePointerInfo(), MachinePointerInfo());
      MemOpChains.push_back(Memcpy);
    } else {
      assert(VA.isMemLoc() && "Argument not register or memory");

      // Work out the address of the stack slot.  Unpromoted ints and
      // floats are passed as right-justified 8-byte values.
      if (!StackPtr.getNode())
        StackPtr = DAG.getCopyFromReg(Chain, DL, Xtensa::SP, PtrVT);
      unsigned Offset = VA.getLocMemOffset();
      SDValue Address = DAG.getNode(ISD::ADD, DL, PtrVT, StackPtr,
                                    DAG.getIntPtrConstant(Offset, DL));

      // Emit the store.
      MemOpChains.push_back(
          DAG.getStore(Chain, DL, ArgValue, Address, MachinePointerInfo()));
    }
  }

  // Join the stores, which are independent of one another.
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, MemOpChains);

  // Build a sequence of copy-to-reg nodes, chained and glued together.
  SDValue Glue;
  for (unsigned I = 0, E = RegsToPass.size(); I != E; ++I) {
    unsigned Reg = RegsToPass[I].first;
    Chain = DAG.getCopyToReg(Chain, DL, Reg, RegsToPass[I].second, Glue);
    Glue = Chain.getValue(1);
  }
  std::string name;
  unsigned char TF = 0;

  // Accept direct calls by converting symbolic call addresses to the
  // associated Target* opcodes.
  if (ExternalSymbolSDNode *E = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    name = E->getSymbol();
    TF = E->getTargetFlags();
    if (isPositionIndependent()) {
      report_fatal_error("PIC relocations is not supported");
    } else
      Callee = DAG.getTargetExternalSymbol(E->getSymbol(), PtrVT, TF);
  } else if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    const GlobalValue *GV = G->getGlobal();
    name = GV->getName().str();
  }

  if ((!name.empty()) && isLongCall(name.c_str())) {
    // Create a constant pool entry for the callee address
    XtensaCP::XtensaCPModifier Modifier = XtensaCP::no_modifier;

    XtensaConstantPoolValue *CPV = XtensaConstantPoolSymbol::Create(
        *DAG.getContext(), name.c_str(), 0 /* XtensaCLabelIndex */, false,
        Modifier);

    // Get the address of the callee into a register
    SDValue CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, Align(4), 0, TF);
    SDValue CPWrap = getAddrPCRel(CPAddr, DAG);
    Callee = CPWrap;
  }

  // The first call operand is the chain and the second is the target address.
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add a register mask operand representing the call-preserved registers.
  const TargetRegisterInfo *TRI = Subtarget.getRegisterInfo();
  const uint32_t *Mask = TRI->getCallPreservedMask(MF, CallConv);
  assert(Mask && "Missing call preserved mask for calling convention");
  Ops.push_back(DAG.getRegisterMask(Mask));

  // Add argument registers to the end of the list so that they are
  // known live into the call.
  for (unsigned I = 0, E = RegsToPass.size(); I != E; ++I) {
    unsigned Reg = RegsToPass[I].first;
    Ops.push_back(DAG.getRegister(Reg, RegsToPass[I].second.getValueType()));
  }

  // Glue the call to the argument copies, if any.
  if (Glue.getNode())
    Ops.push_back(Glue);

  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  Chain = DAG.getNode(XtensaISD::CALL, DL, NodeTys, Ops);
  Glue = Chain.getValue(1);

  // Mark the end of the call, which is glued to the call itself.
  Chain = DAG.getCALLSEQ_END(Chain, DAG.getConstant(NumBytes, DL, PtrVT, true),
                             DAG.getConstant(0, DL, PtrVT, true), Glue, DL);
  Glue = Chain.getValue(1);

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RetLocs;
  CCState RetCCInfo(CallConv, IsVarArg, MF, RetLocs, *DAG.getContext());
  RetCCInfo.AnalyzeCallResult(Ins, RetCC_Xtensa);

  // Copy all of the result registers out of their specified physreg.
  for (unsigned I = 0, E = RetLocs.size(); I != E; ++I) {
    CCValAssign &VA = RetLocs[I];

    // Copy the value out, gluing the copy to the end of the call sequence.
    unsigned Reg = VA.getLocReg();
    SDValue RetValue = DAG.getCopyFromReg(Chain, DL, Reg, VA.getLocVT(), Glue);
    Chain = RetValue.getValue(1);
    Glue = RetValue.getValue(2);

    InVals.push_back(RetValue);
  }
  return Chain;
}

bool XtensaTargetLowering::CanLowerReturn(
    CallingConv::ID CallConv, MachineFunction &MF, bool IsVarArg,
    const SmallVectorImpl<ISD::OutputArg> &Outs, LLVMContext &Context) const {
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, MF, RVLocs, Context);
  return CCInfo.CheckReturn(Outs, RetCC_Xtensa);
}

SDValue
XtensaTargetLowering::LowerReturn(SDValue Chain, CallingConv::ID CallConv,
                                  bool IsVarArg,
                                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                                  const SmallVectorImpl<SDValue> &OutVals,
                                  const SDLoc &DL, SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();

  // Assign locations to each returned value.
  SmallVector<CCValAssign, 16> RetLocs;
  CCState RetCCInfo(CallConv, IsVarArg, MF, RetLocs, *DAG.getContext());
  RetCCInfo.AnalyzeReturn(Outs, RetCC_Xtensa);

  SDValue Glue;
  // Quick exit for void returns
  if (RetLocs.empty())
    return DAG.getNode(XtensaISD::RET, DL, MVT::Other, Chain);

  // Copy the result values into the output registers.
  SmallVector<SDValue, 4> RetOps;
  RetOps.push_back(Chain);
  for (unsigned I = 0, E = RetLocs.size(); I != E; ++I) {
    CCValAssign &VA = RetLocs[I];
    SDValue RetValue = OutVals[I];

    // Make the return register live on exit.
    assert(VA.isRegLoc() && "Can only return in registers!");

    // Chain and glue the copies together.
    unsigned Register = VA.getLocReg();
    Chain = DAG.getCopyToReg(Chain, DL, Register, RetValue, Glue);
    Glue = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(Register, VA.getLocVT()));
  }

  // Update chain and glue.
  RetOps[0] = Chain;
  if (Glue.getNode())
    RetOps.push_back(Glue);

  return DAG.getNode(XtensaISD::RET, DL, MVT::Other, RetOps);
}

static unsigned getBranchOpcode(ISD::CondCode Cond) {
  switch (Cond) {
  case ISD::SETEQ:
    return Xtensa::BEQ;
  case ISD::SETNE:
    return Xtensa::BNE;
  case ISD::SETLT:
    return Xtensa::BLT;
  case ISD::SETLE:
    return Xtensa::BGE;
  case ISD::SETGT:
    return Xtensa::BLT;
  case ISD::SETGE:
    return Xtensa::BGE;
  case ISD::SETULT:
    return Xtensa::BLTU;
  case ISD::SETULE:
    return Xtensa::BGEU;
  case ISD::SETUGT:
    return Xtensa::BLTU;
  case ISD::SETUGE:
    return Xtensa::BGEU;
  default:
    llvm_unreachable("Unknown branch kind");
  }
}

SDValue XtensaTargetLowering::LowerSELECT_CC(SDValue Op,
                                             SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = Op.getOperand(0).getValueType();
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue TrueValue = Op.getOperand(2);
  SDValue FalseValue = Op.getOperand(3);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op->getOperand(4))->get();

  unsigned BrOpcode = getBranchOpcode(CC);
  SDValue TargetCC = DAG.getConstant(BrOpcode, DL, MVT::i32);

  return DAG.getNode(XtensaISD::SELECT_CC, DL, Ty, LHS, RHS, TrueValue,
                     FalseValue, TargetCC);
}

SDValue XtensaTargetLowering::LowerRETURNADDR(SDValue Op,
                                              SelectionDAG &DAG) const {
  // This nodes represent llvm.returnaddress on the DAG.
  // It takes one operand, the index of the return address to return.
  // An index of zero corresponds to the current function's return address.
  // An index of one to the parent's return address, and so on.
  // Depths > 0 not supported yet!
  if (Op.getConstantOperandVal(0) != 0)
    return SDValue();

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  EVT VT = Op.getValueType();
  MFI.setReturnAddressIsTaken(true);

  // Return RA, which contains the return address. Mark it an implicit
  // live-in.
  Register RA = MF.addLiveIn(Xtensa::A0, getRegClassFor(MVT::i32));
  return DAG.getCopyFromReg(DAG.getEntryNode(), SDLoc(Op), RA, VT);
}

SDValue XtensaTargetLowering::LowerImmediate(SDValue Op,
                                             SelectionDAG &DAG) const {
  const ConstantSDNode *CN = cast<ConstantSDNode>(Op);
  SDLoc DL(CN);
  APInt APVal = CN->getAPIntValue();
  int64_t Value = APVal.getSExtValue();
  if (Op.getValueType() == MVT::i32) {
    // Check if use node maybe lowered to the MOVI instruction
    if (Value > -2048 && Value <= 2047)
      return Op;
    // Check if use node maybe lowered to the ADDMI instruction
    SDNode &OpNode = *Op.getNode();
    if ((OpNode.hasOneUse() && OpNode.use_begin()->getOpcode() == ISD::ADD) &&
        isShiftedInt<16, 8>(Value))
      return Op;
    Type *Ty = Type::getInt32Ty(*DAG.getContext());
    Constant *CV = ConstantInt::get(Ty, Value);
    SDValue CP = DAG.getConstantPool(CV, MVT::i32);
    return CP;
  }
  return Op;
}

SDValue XtensaTargetLowering::LowerGlobalAddress(SDValue Op,
                                                 SelectionDAG &DAG) const {
  const GlobalAddressSDNode *G = cast<GlobalAddressSDNode>(Op);
  SDLoc DL(Op);
  auto PtrVT = Op.getValueType();
  const GlobalValue *GV = G->getGlobal();

  SDValue CPAddr = DAG.getTargetConstantPool(GV, PtrVT, Align(4));
  SDValue CPWrap = getAddrPCRel(CPAddr, DAG);

  return CPWrap;
}

SDValue XtensaTargetLowering::LowerBlockAddress(SDValue Op,
                                                SelectionDAG &DAG) const {
  BlockAddressSDNode *Node = cast<BlockAddressSDNode>(Op);
  const BlockAddress *BA = Node->getBlockAddress();
  EVT PtrVT = Op.getValueType();

  XtensaConstantPoolValue *CPV =
      XtensaConstantPoolConstant::Create(BA, 0, XtensaCP::CPBlockAddress);
  SDValue CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, Align(4));
  SDValue CPWrap = getAddrPCRel(CPAddr, DAG);

  return CPWrap;
}

SDValue XtensaTargetLowering::LowerBR_JT(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  SDValue Table = Op.getOperand(1);
  SDValue Index = Op.getOperand(2);
  SDLoc DL(Op);
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Table);
  MachineFunction &MF = DAG.getMachineFunction();
  const MachineJumpTableInfo *MJTI = MF.getJumpTableInfo();
  SDValue TargetJT = DAG.getTargetJumpTable(JT->getIndex(), MVT::i32);
  const DataLayout &TD = DAG.getDataLayout();
  EVT PtrVT = Table.getValueType();
  unsigned EntrySize = MJTI->getEntrySize(TD);

  assert((MJTI->getEntrySize(TD) == 4) && "Unsupported jump-table entry size");

  Index = DAG.getNode(
      ISD::SHL, DL, Index.getValueType(), Index,
      DAG.getConstant(Log2_32(EntrySize), DL, Index.getValueType()));

  SDValue Addr = DAG.getNode(ISD::ADD, DL, Index.getValueType(), Index, Table);
  SDValue LD =
      DAG.getLoad(PtrVT, DL, Chain, Addr,
                  MachinePointerInfo::getJumpTable(DAG.getMachineFunction()));

  return DAG.getNode(XtensaISD::BR_JT, DL, MVT::Other, LD.getValue(1), LD,
                     TargetJT);
}

SDValue XtensaTargetLowering::LowerJumpTable(SDValue Op,
                                             SelectionDAG &DAG) const {
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Op);
  EVT PtrVT = Op.getValueType();

  // Create a constant pool entry for the callee address
  XtensaConstantPoolValue *CPV =
      XtensaConstantPoolJumpTable::Create(*DAG.getContext(), JT->getIndex());

  // Get the address of the callee into a register
  SDValue CPAddr = DAG.getTargetConstantPool(CPV, PtrVT, Align(4));

  return getAddrPCRel(CPAddr, DAG);
}

SDValue XtensaTargetLowering::getAddrPCRel(SDValue Op,
                                           SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT Ty = Op.getValueType();
  return DAG.getNode(XtensaISD::PCREL_WRAPPER, DL, Ty, Op);
}

SDValue XtensaTargetLowering::LowerConstantPool(SDValue Op,
                                                SelectionDAG &DAG) const {
  EVT PtrVT = Op.getValueType();
  ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
  SDValue Result;

  if (!CP->isMachineConstantPoolEntry()) {
    Result = DAG.getTargetConstantPool(CP->getConstVal(), PtrVT, CP->getAlign(),
                                       CP->getOffset());
  } else {
    report_fatal_error("This constantpool type is not supported yet");
  }

  return getAddrPCRel(Result, DAG);
}

SDValue XtensaTargetLowering::LowerSTACKSAVE(SDValue Op,
                                             SelectionDAG &DAG) const {
  return DAG.getCopyFromReg(Op.getOperand(0), SDLoc(Op), Xtensa::SP,
                            Op.getValueType());
}

SDValue XtensaTargetLowering::LowerSTACKRESTORE(SDValue Op,
                                                SelectionDAG &DAG) const {
  return DAG.getCopyToReg(Op.getOperand(0), SDLoc(Op), Xtensa::SP,
                          Op.getOperand(1));
}

SDValue XtensaTargetLowering::LowerFRAMEADDR(SDValue Op,
                                             SelectionDAG &DAG) const {
  // This nodes represent llvm.frameaddress on the DAG.
  // It takes one operand, the index of the frame address to return.
  // An index of zero corresponds to the current function's frame address.
  // An index of one to the parent's frame address, and so on.
  // Depths > 0 not supported yet!
  if (Op.getConstantOperandVal(0) != 0)
    return SDValue();

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  MFI.setFrameAddressIsTaken(true);
  EVT VT = Op.getValueType();
  SDLoc DL(Op);

  Register FrameRegister = Subtarget.getRegisterInfo()->getFrameRegister(MF);
  SDValue FrameAddr =
      DAG.getCopyFromReg(DAG.getEntryNode(), DL, FrameRegister, VT);
  return FrameAddr;
}

SDValue XtensaTargetLowering::LowerDYNAMIC_STACKALLOC(SDValue Op,
                                                      SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0); // Legalize the chain.
  SDValue Size = Op.getOperand(1);  // Legalize the size.
  EVT VT = Size->getValueType(0);
  SDLoc DL(Op);

  // Round up Size to 32
  SDValue SizeTmp =
      DAG.getNode(ISD::ADD, DL, VT, Size, DAG.getConstant(31, DL, MVT::i32));
  SDValue SizeRoundUp = DAG.getNode(ISD::AND, DL, VT, SizeTmp,
                                    DAG.getConstant(~31, DL, MVT::i32));

  unsigned SPReg = Xtensa::SP;
  SDValue SP = DAG.getCopyFromReg(Chain, DL, SPReg, VT);
  SDValue NewSP = DAG.getNode(ISD::SUB, DL, VT, SP, SizeRoundUp); // Value
  Chain = DAG.getCopyToReg(SP.getValue(1), DL, SPReg, NewSP); // Output chain

  SDValue NewVal = DAG.getCopyFromReg(Chain, DL, SPReg, MVT::i32);
  Chain = NewVal.getValue(1);

  SDValue Ops[2] = {NewVal, Chain};
  return DAG.getMergeValues(Ops, DL);
}

SDValue XtensaTargetLowering::LowerVASTART(SDValue Op,
                                           SelectionDAG &DAG) const {
  MachineFunction &MF = DAG.getMachineFunction();
  XtensaMachineFunctionInfo *XtensaFI = MF.getInfo<XtensaMachineFunctionInfo>();
  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  SDLoc DL(Op);

  SDValue Chain = Op.getOperand(0);
  SDValue Addr = Op.getOperand(1);

  // typedef struct __va_list_tag {
  //   int32_t *__va_stk; /* Initialized to point  to the position of the
  //                       * first argument in memory offset to account for
  //                       the
  //                       * arguments passed in registers and to account for
  //                       * the size of the argument registers not being
  //                       16-byte
  //                       * aligned.  E.G., there are 6 argument registers
  //                       * of 4 bytes each, but we want the __va_ndx for the
  //                       * first stack argument to have the maximal
  //                       * alignment of 16 bytes, so we offset the __va_stk
  //                       address by
  //                       * 32 bytes so that __va_stk[32] references the
  //                       first
  //                       * argument on the stack.
  //                       */
  //   int32_t  *__va_reg; /* Points to a stack-allocated region holding the
  //                        * contents
  //                        * of the incoming argument registers
  //                        */
  //   int32_t __va_ndx;   /* Index initialized to the position of the first
  //                        * unnamed (variable) argument.  This same index is
  //                        also
  //                        * used to address the arguments passed in memory.
  //                       */
  //  } __va_list_tag[1];

  SDValue ArgAR;
  SDValue OverflowPtrAdvance;
  SDValue StackOffsetFI =
      DAG.getFrameIndex(XtensaFI->getVarArgsStackOffset(), PtrVT);

  if (XtensaFI->getVarArgsFirstGPR() < 8) {
    ArgAR =
        DAG.getConstant(XtensaFI->getVarArgsFirstGPR() * 4 - 8, DL, MVT::i32);
    OverflowPtrAdvance = DAG.getConstant(32, DL, PtrVT);
  } else {
    OverflowPtrAdvance = DAG.getNode(ISD::AND, DL, PtrVT, StackOffsetFI,
                                     DAG.getConstant(0xf, DL, PtrVT));
    OverflowPtrAdvance = DAG.getNode(ISD::ADD, DL, PtrVT, OverflowPtrAdvance,
                                     DAG.getConstant(32, DL, PtrVT));
    ArgAR = OverflowPtrAdvance;
  }

  SDValue FR = DAG.getFrameIndex(XtensaFI->getVarArgsFrameIndex(), PtrVT);

  uint64_t FrameOffset = PtrVT.getSizeInBits() / 8;
  SDValue ConstFrameOffset1 = DAG.getConstant(FrameOffset, DL, PtrVT);
  SDValue ConstFrameOffset2 = DAG.getConstant(FrameOffset * 2, DL, PtrVT);

  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();

  // Store first word : arguments given in stack  (__va_stk)
  // Advance Argument Overflow pointer down, lest it will point to start
  // after register argument va_arg finished
  SDValue StackOffsetFICorr =
      DAG.getNode(ISD::SUB, DL, PtrVT, StackOffsetFI, OverflowPtrAdvance);
  SDValue firstStore =
      DAG.getStore(Chain, DL, StackOffsetFICorr, Addr, MachinePointerInfo(SV));

  uint64_t nextOffset = FrameOffset;
  SDValue nextPtr = DAG.getNode(ISD::ADD, DL, PtrVT, Addr, ConstFrameOffset1);

  // Store second word : arguments given on registers  (__va_reg)
  SDValue FRAdvance =
      DAG.getConstant(XtensaFI->getVarArgsFirstGPR() * 4 - 8, DL, PtrVT);
  SDValue FRDecr = DAG.getNode(ISD::SUB, DL, PtrVT, FR, FRAdvance);
  SDValue secondStore = DAG.getStore(firstStore, DL, FRDecr, nextPtr,
                                     MachinePointerInfo(SV, nextOffset));
  nextOffset += FrameOffset;
  nextPtr = DAG.getNode(ISD::ADD, DL, PtrVT, Addr, ConstFrameOffset2);

  // Store first word : number of int regs  (__va_ndx)
  return DAG.getStore(secondStore, DL, ArgAR, nextPtr,
                      MachinePointerInfo(SV, nextOffset));
}

SDValue XtensaTargetLowering::LowerVACOPY(SDValue Op, SelectionDAG &DAG) const {
  unsigned VAListSize = getVaListSizeInBits(DAG.getDataLayout());
  return DAG.getMemcpy(
      Op.getOperand(0), Op, Op.getOperand(1), Op.getOperand(2),
      DAG.getConstant(VAListSize, SDLoc(Op), MVT::i32), Align(4),
      /*isVolatile=*/false, /*AlwaysInline=*/false,
      /*CI=*/nullptr, std::nullopt, MachinePointerInfo(), MachinePointerInfo());
}

SDValue XtensaTargetLowering::LowerVAARG(SDValue Op, SelectionDAG &DAG) const {
  SDNode *Node = Op.getNode();
  EVT VT = Node->getValueType(0);
  EVT PtrVT = getPointerTy(DAG.getDataLayout());
  SDValue InChain = Node->getOperand(0);
  SDValue VAListPtr = Node->getOperand(1);
  const Value *SV = cast<SrcValueSDNode>(Node->getOperand(2))->getValue();
  SDLoc DL(Node);
#if 1
#if 0
  /// VAARG - VAARG has four operands: an input chain, a pointer, a SRCVALUE,
  /// and the alignment. It returns a pair of values: the vaarg value and a
  /// new chain.
  VAARG,

  /// VACOPY - VACOPY has 5 operands: an input chain, a destination pointer,
  /// a source pointer, a SRCVALUE for the destination, and a SRCVALUE for the
  /// source.
  VACOPY,

  /// VAEND, VASTART - VAEND and VASTART have three operands: an input chain,
  /// pointer, and a SRCVALUE.
  VAEND,
  VASTART,

#endif
  auto &TD = DAG.getDataLayout();
  Align ArgAlignment = TD.getPrefTypeAlign(VT.getTypeForEVT(*DAG.getContext()));
  unsigned ArgAlignInBytes = ArgAlignment.value();
  unsigned ArgSizeInBytes =
      TD.getTypeAllocSize(VT.getTypeForEVT(*DAG.getContext()));
  unsigned VASizeInBytes = (ArgSizeInBytes + 3) & 0x3;

  // areas
  // va_stk
  SDValue OverflowArea =
      DAG.getLoad(MVT::i32, DL, InChain, VAListPtr, MachinePointerInfo());
  InChain = OverflowArea.getValue(1);

  SDValue RegSaveAreaPtr = DAG.getNode(ISD::ADD, DL, PtrVT, VAListPtr,
                                       DAG.getConstant(4, DL, MVT::i32));

  // va_reg
  SDValue RegSaveArea =
      DAG.getLoad(MVT::i32, DL, InChain, RegSaveAreaPtr, MachinePointerInfo());
  InChain = RegSaveArea.getValue(1);

  // va_ndx
  SDValue ARAreaPtr = DAG.getNode(ISD::ADD, DL, PtrVT, RegSaveAreaPtr,
                                  DAG.getConstant(4, DL, MVT::i32));

  SDValue ARIndex =
      DAG.getLoad(MVT::i32, DL, InChain, ARAreaPtr, MachinePointerInfo());
  InChain = ARIndex.getValue(1);

  SDValue OrigIndex = ARIndex;

  if (ArgAlignInBytes > 4) {
    OrigIndex = DAG.getNode(ISD::ADD, DL, PtrVT, OrigIndex,
                            DAG.getConstant(ArgAlignInBytes - 1, DL, MVT::i32));
    OrigIndex = DAG.getNode(ISD::AND, DL, PtrVT, OrigIndex,
                            DAG.getConstant(-ArgAlignInBytes, DL, MVT::i32));
  }

  ARIndex = DAG.getNode(ISD::ADD, DL, PtrVT, OrigIndex,
                        DAG.getConstant(VASizeInBytes, DL, MVT::i32));

  SDValue CC = DAG.getSetCC(DL, MVT::i32, OrigIndex,
                            DAG.getConstant(6 * 4, DL, MVT::i32), ISD::SETLE);

  SDValue StkIndex =
      DAG.getNode(ISD::ADD, DL, PtrVT, ARIndex,
                  DAG.getConstant(32 + VASizeInBytes, DL, MVT::i32));

  CC = DAG.getSetCC(DL, MVT::i32, ARIndex, DAG.getConstant(6 * 4, DL, MVT::i32),
                    ISD::SETLE);

  SDValue Array =
      DAG.getNode(ISD::SELECT, DL, MVT::i32, CC, RegSaveArea, OverflowArea);

  ARIndex = DAG.getNode(ISD::SELECT, DL, MVT::i32, CC, ARIndex, StkIndex);

  CC = DAG.getSetCC(DL, MVT::i32, ARIndex, DAG.getConstant(6 * 4, DL, MVT::i32),
                    ISD::SETLE);

  SDValue ARIndexStore =
      DAG.getStore(InChain, DL, ARIndex, ARAreaPtr, MachinePointerInfo(SV));
  InChain = ARIndexStore;

  SDValue Addr = DAG.getNode(ISD::SUB, DL, PtrVT, ARIndex,
                             DAG.getConstant(VASizeInBytes, DL, MVT::i32));

  Addr = DAG.getNode(ISD::ADD, DL, PtrVT, Array, Addr);

  return DAG.getLoad(VT, DL, InChain, Addr, MachinePointerInfo());

#if 0
  //////////////////////////////////////////////
  // va_stk
  SDValue GprIndex = DAG.getExtLoad(ISD::ZEXTLOAD, dl, MVT::i32, InChain,
                                    VAListPtr, MachinePointerInfo(SV), MVT::i8);
  InChain = GprIndex.getValue(1);

  if (VT == MVT::i64) {
    // Check if GprIndex is even
    SDValue GprAnd = DAG.getNode(ISD::AND, dl, MVT::i32, GprIndex,
                                 DAG.getConstant(1, dl, MVT::i32));
    SDValue CC64 = DAG.getSetCC(dl, MVT::i32, GprAnd,
                                DAG.getConstant(0, dl, MVT::i32), ISD::SETNE);
    SDValue GprIndexPlusOne = DAG.getNode(ISD::ADD, dl, MVT::i32, GprIndex,
                                          DAG.getConstant(1, dl, MVT::i32));
    // Align GprIndex to be even if it isn't
    GprIndex =
        DAG.getNode(ISD::SELECT, dl, MVT::i32, CC64, GprIndexPlusOne, GprIndex);
  }

  // fpr index is 1 byte after gpr
  SDValue FprPtr = DAG.getNode(ISD::ADD, dl, PtrVT, VAListPtr,
                               DAG.getConstant(1, dl, MVT::i32));

  // fpr
  SDValue FprIndex = DAG.getExtLoad(ISD::ZEXTLOAD, dl, MVT::i32, InChain,
                                    FprPtr, MachinePointerInfo(SV), MVT::i8);
  InChain = FprIndex.getValue(1);

  SDValue RegSaveAreaPtr = DAG.getNode(ISD::ADD, dl, PtrVT, VAListPtr,
                                       DAG.getConstant(8, dl, MVT::i32));

  SDValue OverflowAreaPtr = DAG.getNode(ISD::ADD, dl, PtrVT, VAListPtr,
                                        DAG.getConstant(4, dl, MVT::i32));

  // areas
  SDValue OverflowArea =
      DAG.getLoad(MVT::i32, dl, InChain, OverflowAreaPtr, MachinePointerInfo());
  InChain = OverflowArea.getValue(1);

  SDValue RegSaveArea =
      DAG.getLoad(MVT::i32, dl, InChain, RegSaveAreaPtr, MachinePointerInfo());
  InChain = RegSaveArea.getValue(1);

  // select overflow_area if index > 8
  SDValue CC = DAG.getSetCC(dl, MVT::i32, VT.isInteger() ? GprIndex : FprIndex,
                            DAG.getConstant(8, dl, MVT::i32), ISD::SETLT);

  // adjustment constant gpr_index * 4/8
  SDValue RegConstant =
      DAG.getNode(ISD::MUL, dl, MVT::i32, VT.isInteger() ? GprIndex : FprIndex,
                  DAG.getConstant(VT.isInteger() ? 4 : 8, dl, MVT::i32));

  // OurReg = RegSaveArea + RegConstant
  SDValue OurReg = DAG.getNode(ISD::ADD, dl, PtrVT, RegSaveArea, RegConstant);

  // Floating types are 32 bytes into RegSaveArea
  if (VT.isFloatingPoint())
    OurReg = DAG.getNode(ISD::ADD, dl, PtrVT, OurReg,
                         DAG.getConstant(32, dl, MVT::i32));

  // increase {f,g}pr_index by 1 (or 2 if VT is i64)
  SDValue IndexPlus1 =
      DAG.getNode(ISD::ADD, dl, MVT::i32, VT.isInteger() ? GprIndex : FprIndex,
                  DAG.getConstant(VT == MVT::i64 ? 2 : 1, dl, MVT::i32));

  InChain = DAG.getTruncStore(InChain, dl, IndexPlus1,
                              VT.isInteger() ? VAListPtr : FprPtr,
                              MachinePointerInfo(SV), MVT::i8);

  // determine if we should load from reg_save_area or overflow_area
  SDValue Result =
      DAG.getNode(ISD::SELECT, dl, PtrVT, CC, OurReg, OverflowArea);

  // increase overflow_area by 4/8 if gpr/fpr > 8
  SDValue OverflowAreaPlusN =
      DAG.getNode(ISD::ADD, dl, PtrVT, OverflowArea,
                  DAG.getConstant(VT.isInteger() ? 4 : 8, dl, MVT::i32));

  InChain = DAG.getTruncStore(InChain, dl, OverflowArea,
                              Ove OverflowArea =
                                  DAG.getNode(ISD::SELECT, dl, MVT::i32, CC,
                                              OverflowArea, OverflowAreaPlusN);
                              rflowAreaPtr, MachinePointerInfo(), MVT::i32);

  return DAG.getLoad(VT, dl, InChain, Result, MachinePointerInfo());
#endif
#else
  // gpr_index
  SDValue GprIndex = DAG.getExtLoad(ISD::ZEXTLOAD, dl, MVT::i32, InChain,
                                    VAListPtr, MachinePointerInfo(SV), MVT::i8);
  InChain = GprIndex.getValue(1);

  if (VT == MVT::i64) {
    // Check if GprIndex is even
    SDValue GprAnd = DAG.getNode(ISD::AND, dl, MVT::i32, GprIndex,
                                 DAG.getConstant(1, dl, MVT::i32));
    SDValue CC64 = DAG.getSetCC(dl, MVT::i32, GprAnd,
                                DAG.getConstant(0, dl, MVT::i32), ISD::SETNE);
    SDValue GprIndexPlusOne = DAG.getNode(ISD::ADD, dl, MVT::i32, GprIndex,
                                          DAG.getConstant(1, dl, MVT::i32));
    // Align GprIndex to be even if it isn't
    GprIndex =
        DAG.getNode(ISD::SELECT, dl, MVT::i32, CC64, GprIndexPlusOne, GprIndex);
  }

  // fpr index is 1 byte after gpr
  SDValue FprPtr = DAG.getNode(ISD::ADD, dl, PtrVT, VAListPtr,
                               DAG.getConstant(1, dl, MVT::i32));

  // fpr
  SDValue FprIndex = DAG.getExtLoad(ISD::ZEXTLOAD, dl, MVT::i32, InChain,
                                    FprPtr, MachinePointerInfo(SV), MVT::i8);
  InChain = FprIndex.getValue(1);

  SDValue RegSaveAreaPtr = DAG.getNode(ISD::ADD, dl, PtrVT, VAListPtr,
                                       DAG.getConstant(8, dl, MVT::i32));

  SDValue OverflowAreaPtr = DAG.getNode(ISD::ADD, dl, PtrVT, VAListPtr,
                                        DAG.getConstant(4, dl, MVT::i32));

  // areas
  SDValue OverflowArea =
      DAG.getLoad(MVT::i32, dl, InChain, OverflowAreaPtr, MachinePointerInfo());
  InChain = OverflowArea.getValue(1);

  SDValue RegSaveArea =
      DAG.getLoad(MVT::i32, dl, InChain, RegSaveAreaPtr, MachinePointerInfo());
  InChain = RegSaveArea.getValue(1);

  // select overflow_area if index > 8
  SDValue CC = DAG.getSetCC(dl, MVT::i32, VT.isInteger() ? GprIndex : FprIndex,
                            DAG.getConstant(8, dl, MVT::i32), ISD::SETLT);

  // adjustment constant gpr_index * 4/8
  SDValue RegConstant =
      DAG.getNode(ISD::MUL, dl, MVT::i32, VT.isInteger() ? GprIndex : FprIndex,
                  DAG.getConstant(VT.isInteger() ? 4 : 8, dl, MVT::i32));

  // OurReg = RegSaveArea + RegConstant
  SDValue OurReg = DAG.getNode(ISD::ADD, dl, PtrVT, RegSaveArea, RegConstant);

  // Floating types are 32 bytes into RegSaveArea
  if (VT.isFloatingPoint())
    OurReg = DAG.getNode(ISD::ADD, dl, PtrVT, OurReg,
                         DAG.getConstant(32, dl, MVT::i32));

  // increase {f,g}pr_index by 1 (or 2 if VT is i64)
  SDValue IndexPlus1 =
      DAG.getNode(ISD::ADD, dl, MVT::i32, VT.isInteger() ? GprIndex : FprIndex,
                  DAG.getConstant(VT == MVT::i64 ? 2 : 1, dl, MVT::i32));

  InChain = DAG.getTruncStore(InChain, dl, IndexPlus1,
                              VT.isInteger() ? VAListPtr : FprPtr,
                              MachinePointerInfo(SV), MVT::i8);

  // determine if we should load from reg_save_area or overflow_area
  SDValue Result =
      DAG.getNode(ISD::SELECT, dl, PtrVT, CC, OurReg, OverflowArea);

  // increase overflow_area by 4/8 if gpr/fpr > 8
  SDValue OverflowAreaPlusN =
      DAG.getNode(ISD::ADD, dl, PtrVT, OverflowArea,
                  DAG.getConstant(VT.isInteger() ? 4 : 8, dl, MVT::i32));

  InChain = DAG.getTruncStore(InChain, dl, OverflowArea,
                              Ove OverflowArea =
                                  DAG.getNode(ISD::SELECT, dl, MVT::i32, CC,
                                              OverflowArea, OverflowAreaPlusN);
                              rflowAreaPtr, MachinePointerInfo(), MVT::i32);

  return DAG.getLoad(VT, dl, InChain, Result, MachinePointerInfo());
#endif
}

#if 0
Address XtensaABIInfo::EmitVAArg(CodeGenFunction &CGF, Address VAListAddr,
                                 QualType Ty) const {
  // The va_list structure memory layout:
  // struct __va_list_tag {
  //   int32_t *va_stk;
  //   int32_t *va_reg;
  //   int32_t va_ndx;
  // };
  CGBuilderTy &Builder = CGF.Builder;

  Address OverflowAreaPtr = Builder.CreateStructGEP(VAListAddr, 0, "__va_stk");
  Address OverflowArea = Address(Builder.CreateLoad(OverflowAreaPtr, ""),
                                 CGF.Int32Ty, CharUnits::fromQuantity(4));
  Address RegSaveAreaPtr = Builder.CreateStructGEP(VAListAddr, 1, "__va_reg");
  Address RegSaveArea = Address(Builder.CreateLoad(RegSaveAreaPtr, ""),
                                CGF.Int32Ty, CharUnits::fromQuantity(4));
  Address ARAreaPtr = Builder.CreateStructGEP(VAListAddr, 2, "__va_ndx");
  llvm::Value *ARIndex = Builder.CreateLoad(ARAreaPtr, "");

  ARIndex = Builder.CreateLShr(ARIndex, Builder.getInt32(2));

  unsigned Align = getContext().getTypeAlign(Ty) / 32;
  unsigned Size = (getContext().getTypeSize(Ty) + 31) / 32;

  if (Align > 1) {
    ARIndex = Builder.CreateAdd(ARIndex, Builder.getInt32(Align - 1));
    ARIndex =
        Builder.CreateAnd(ARIndex, Builder.getInt32((uint32_t) ~(Align - 1)));
  }

  llvm::Value *ARIndexNext = Builder.CreateAdd(ARIndex, Builder.getInt32(Size));
  Builder.CreateStore(Builder.CreateShl(ARIndexNext, Builder.getInt32(2)),
                      ARAreaPtr);

  const unsigned OverflowLimit = 6;
  llvm::Value *CC = Builder.CreateICmpULE(
      ARIndexNext, Builder.getInt32(OverflowLimit), "cond");

  llvm::BasicBlock *UsingRegSaveArea =
      CGF.createBasicBlock("using_regsavearea");
  llvm::BasicBlock *UsingOverflow = CGF.createBasicBlock("using_overflow");
  llvm::BasicBlock *Cont = CGF.createBasicBlock("cont");

  Builder.CreateCondBr(CC, UsingRegSaveArea, UsingOverflow);

  llvm::Type *DirectTy = CGF.ConvertType(Ty);

  // Case 1: consume registers.
  Address RegAddr = Address::invalid();
  {
    CGF.EmitBlock(UsingRegSaveArea);

    CharUnits RegSize = CharUnits::fromQuantity(4);
    RegSaveArea =
        Address(Builder.CreateInBoundsGEP(CGF.Int32Ty, RegSaveArea.getPointer(),
                                          ARIndex),
                CGF.Int32Ty, RegSaveArea.getAlignment().alignmentOfArrayElement(RegSize));
    RegAddr = RegSaveArea.withElementType(DirectTy);
    CGF.EmitBranch(Cont);
  }

  // Case 2: consume space in the overflow area.
  Address MemAddr = Address::invalid();
  {
    CGF.EmitBlock(UsingOverflow);
    llvm::Value *CC1 = Builder.CreateICmpULE(
        ARIndex, Builder.getInt32(OverflowLimit), "cond_overflow");

    llvm::Value *ARIndexOff = Builder.CreateSelect(
        CC1, Builder.CreateSub(Builder.getInt32(8), ARIndex),
        Builder.getInt32(0));

    llvm::Value *ARIndexCorr = Builder.CreateAdd(ARIndex, ARIndexOff);
    llvm::Value *ARIndexNextCorr = Builder.CreateAdd(ARIndexNext, ARIndexOff);
    Builder.CreateStore(Builder.CreateShl(ARIndexNextCorr, Builder.getInt32(2)),
                        ARAreaPtr);

    CharUnits RegSize = CharUnits::fromQuantity(4);
    OverflowArea =
        Address(Builder.CreateInBoundsGEP(
                    CGF.Int32Ty, OverflowArea.getPointer(), ARIndexCorr),
                CGF.Int32Ty, OverflowArea.getAlignment().alignmentOfArrayElement(RegSize));
    MemAddr = OverflowArea.withElementType(DirectTy);
    CGF.EmitBranch(Cont);
  }

  CGF.EmitBlock(Cont);

  // Merge the cases with a phi.
  Address Result =
      emitMergePHI(CGF, RegAddr, UsingRegSaveArea, MemAddr, UsingOverflow, "");

  return Result;
}

#endif
SDValue XtensaTargetLowering::LowerShiftLeftParts(SDValue Op,
                                                  SelectionDAG &DAG) const {
  SDLoc DL(Op);
  MVT VT = MVT::i32;
  SDValue Lo = Op.getOperand(0), Hi = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);

  // if Shamt - register size < 0: // Shamt < register size
  //   Lo = Lo << Shamt
  //   Hi = (Hi << Shamt) | (Lo >>u (register size - Shamt))
  // else:
  //   Lo = 0
  //   Hi = Lo << (Shamt - register size)

  SDValue MinusRegisterSize = DAG.getConstant(-32, DL, VT);
  SDValue ShamtMinusRegisterSize =
      DAG.getNode(ISD::ADD, DL, VT, Shamt, MinusRegisterSize);

  SDValue LoTrue = DAG.getNode(ISD::SHL, DL, VT, Lo, Shamt);
  SDValue HiTrue = DAG.getNode(XtensaISD::SRCL, DL, VT, Hi, Lo, Shamt);
  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue HiFalse = DAG.getNode(ISD::SHL, DL, VT, Lo, ShamtMinusRegisterSize);

  SDValue Cond = DAG.getSetCC(DL, VT, ShamtMinusRegisterSize, Zero, ISD::SETLT);
  Lo = DAG.getNode(ISD::SELECT, DL, VT, Cond, LoTrue, Zero);
  Hi = DAG.getNode(ISD::SELECT, DL, VT, Cond, HiTrue, HiFalse);

  return DAG.getMergeValues({Lo, Hi}, DL);
}

SDValue XtensaTargetLowering::LowerShiftRightParts(SDValue Op,
                                                   SelectionDAG &DAG,
                                                   bool IsSRA) const {
  SDLoc DL(Op);
  SDValue Lo = Op.getOperand(0), Hi = Op.getOperand(1);
  SDValue Shamt = Op.getOperand(2);
  MVT VT = MVT::i32;

  // SRA expansion:
  //   if Shamt - register size < 0: // Shamt < register size
  //     Lo = (Lo >>u Shamt) | (Hi << u (register size - Shamt))
  //     Hi = Hi >>s Shamt
  //   else:
  //     Lo = Hi >>s (Shamt - register size);
  //     Hi = Hi >>s (register size - 1)
  //
  // SRL expansion:
  //   if Shamt - register size < 0: // Shamt < register size
  //     Lo = (Lo >>u Shamt) | (Hi << u (register size - Shamt))
  //     Hi = Hi >>u Shamt
  //   else:
  //     Lo = Hi >>u (Shamt - register size);
  //     Hi = 0;

  unsigned ShiftRightOp = IsSRA ? ISD::SRA : ISD::SRL;
  SDValue MinusRegisterSize = DAG.getConstant(-32, DL, VT);
  SDValue RegisterSizeMinus1 = DAG.getConstant(32 - 1, DL, VT);
  SDValue ShamtMinusRegisterSize =
      DAG.getNode(ISD::ADD, DL, VT, Shamt, MinusRegisterSize);

  SDValue LoTrue = DAG.getNode(XtensaISD::SRCR, DL, VT, Hi, Lo, Shamt);
  SDValue HiTrue = DAG.getNode(ShiftRightOp, DL, VT, Hi, Shamt);
  SDValue Zero = DAG.getConstant(0, DL, VT);
  SDValue LoFalse =
      DAG.getNode(ShiftRightOp, DL, VT, Hi, ShamtMinusRegisterSize);
  SDValue HiFalse;

  if (IsSRA) {
    HiFalse = DAG.getNode(ShiftRightOp, DL, VT, Hi, RegisterSizeMinus1);
  } else {
    HiFalse = Zero;
  }

  SDValue Cond = DAG.getSetCC(DL, VT, ShamtMinusRegisterSize, Zero, ISD::SETLT);
  Lo = DAG.getNode(ISD::SELECT, DL, VT, Cond, LoTrue, LoFalse);
  Hi = DAG.getNode(ISD::SELECT, DL, VT, Cond, HiTrue, HiFalse);

  return DAG.getMergeValues({Lo, Hi}, DL);
}

SDValue XtensaTargetLowering::LowerCTPOP(SDValue Op, SelectionDAG &DAG) const {
  auto &TLI = DAG.getTargetLoweringInfo();
  return TLI.expandCTPOP(Op.getNode(), DAG);
}

bool XtensaTargetLowering::decomposeMulByConstant(LLVMContext &Context, EVT VT,
                                                  SDValue C) const {
  APInt Imm;
  unsigned EltSizeInBits;

  if (ISD::isConstantSplatVector(C.getNode(), Imm)) {
    EltSizeInBits = VT.getScalarSizeInBits();
  } else if (VT.isScalarInteger()) {
    EltSizeInBits = VT.getSizeInBits();
    if (auto *ConstNode = dyn_cast<ConstantSDNode>(C.getNode()))
      Imm = ConstNode->getAPIntValue();
    else
      return false;
  } else {
    return false;
  }

  // Omit if data size exceeds.
  if (EltSizeInBits > 32)
    return false;

  // Convert MULT to LSL.
  if (Imm.isPowerOf2() && Imm.isIntN(5))
    return true;

  return false;
}

SDValue XtensaTargetLowering::LowerOperation(SDValue Op,
                                             SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::BR_JT:
    return LowerBR_JT(Op, DAG);
  case ISD::Constant:
    return LowerImmediate(Op, DAG);
  case ISD::RETURNADDR:
    return LowerRETURNADDR(Op, DAG);
  case ISD::GlobalAddress:
    return LowerGlobalAddress(Op, DAG);
  case ISD::BlockAddress:
    return LowerBlockAddress(Op, DAG);
  case ISD::JumpTable:
    return LowerJumpTable(Op, DAG);
  case ISD::CTPOP:
    return LowerCTPOP(Op, DAG);
  case ISD::ConstantPool:
    return LowerConstantPool(Op, DAG);
  case ISD::SELECT_CC:
    return LowerSELECT_CC(Op, DAG);
  case ISD::STACKSAVE:
    return LowerSTACKSAVE(Op, DAG);
  case ISD::STACKRESTORE:
    return LowerSTACKRESTORE(Op, DAG);
  case ISD::FRAMEADDR:
    return LowerFRAMEADDR(Op, DAG);
  case ISD::DYNAMIC_STACKALLOC:
    return LowerDYNAMIC_STACKALLOC(Op, DAG);
  case ISD::VASTART:
    return LowerVASTART(Op, DAG);
  case ISD::VAARG:
    return LowerVAARG(Op, DAG);
  case ISD::VACOPY:
    return LowerVACOPY(Op, DAG);
  case ISD::SHL_PARTS:
    return LowerShiftLeftParts(Op, DAG);
  case ISD::SRA_PARTS:
    return LowerShiftRightParts(Op, DAG, true);
  case ISD::SRL_PARTS:
    return LowerShiftRightParts(Op, DAG, false);
  default:
    report_fatal_error("Unexpected node to lower");
  }
}

const char *XtensaTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  case XtensaISD::BR_JT:
    return "XtensaISD::BR_JT";
  case XtensaISD::CALL:
    return "XtensaISD::CALL";
  case XtensaISD::EXTUI:
    return "XtensaISD::EXTUI";
  case XtensaISD::PCREL_WRAPPER:
    return "XtensaISD::PCREL_WRAPPER";
  case XtensaISD::RET:
    return "XtensaISD::RET";
  case XtensaISD::SELECT_CC:
    return "XtensaISD::SELECT_CC";
  case XtensaISD::SRCL:
    return "XtensaISD::SRCL";
  case XtensaISD::SRCR:
    return "XtensaISD::SRCR";
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Custom insertion
//===----------------------------------------------------------------------===//

MachineBasicBlock *
XtensaTargetLowering::emitSelectCC(MachineInstr &MI,
                                   MachineBasicBlock *MBB) const {
  const TargetInstrInfo &TII = *Subtarget.getInstrInfo();
  DebugLoc DL = MI.getDebugLoc();

  MachineOperand &LHS = MI.getOperand(1);
  MachineOperand &RHS = MI.getOperand(2);
  MachineOperand &TrueValue = MI.getOperand(3);
  MachineOperand &FalseValue = MI.getOperand(4);
  unsigned BrKind = MI.getOperand(5).getImm();

  // To "insert" a SELECT_CC instruction, we actually have to insert
  // CopyMBB and SinkMBB  blocks and add branch to MBB. We build phi
  // operation in SinkMBB like phi (TrueVakue,FalseValue), where TrueValue
  // is passed from MMB and FalseValue is passed from CopyMBB.
  //   MBB
  //   |   \
  //   |   CopyMBB
  //   |   /
  //   SinkMBB
  // The incoming instruction knows the
  // destination vreg to set, the condition code register to branch on, the
  // true/false values to select between, and a branch opcode to use.
  const BasicBlock *LLVM_BB = MBB->getBasicBlock();
  MachineFunction::iterator It = ++MBB->getIterator();

  MachineFunction *F = MBB->getParent();
  MachineBasicBlock *CopyMBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *SinkMBB = F->CreateMachineBasicBlock(LLVM_BB);

  F->insert(It, CopyMBB);
  F->insert(It, SinkMBB);

  // Transfer the remainder of MBB and its successor edges to SinkMBB.
  SinkMBB->splice(SinkMBB->begin(), MBB,
                  std::next(MachineBasicBlock::iterator(MI)), MBB->end());
  SinkMBB->transferSuccessorsAndUpdatePHIs(MBB);

  MBB->addSuccessor(CopyMBB);
  MBB->addSuccessor(SinkMBB);

  BuildMI(MBB, DL, TII.get(BrKind))
      .addReg(LHS.getReg())
      .addReg(RHS.getReg())
      .addMBB(SinkMBB);

  CopyMBB->addSuccessor(SinkMBB);

  //  SinkMBB:
  //   %Result = phi [ %FalseValue, CopyMBB ], [ %TrueValue, MBB ]
  //  ...

  BuildMI(*SinkMBB, SinkMBB->begin(), DL, TII.get(Xtensa::PHI),
          MI.getOperand(0).getReg())
      .addReg(FalseValue.getReg())
      .addMBB(CopyMBB)
      .addReg(TrueValue.getReg())
      .addMBB(MBB);

  MI.eraseFromParent(); // The pseudo instruction is gone now.
  return SinkMBB;
}

MachineBasicBlock *XtensaTargetLowering::EmitInstrWithCustomInserter(
    MachineInstr &MI, MachineBasicBlock *MBB) const {
  DebugLoc DL = MI.getDebugLoc();
  const XtensaInstrInfo &TII = *Subtarget.getInstrInfo();

  switch (MI.getOpcode()) {
  case Xtensa::SELECT:
    return emitSelectCC(MI, MBB);
  case Xtensa::S8I:
  case Xtensa::S16I:
  case Xtensa::S32I:
  case Xtensa::L8UI:
  case Xtensa::L16SI:
  case Xtensa::L16UI:
  case Xtensa::L32I: {
    // Insert memory wait instruction "memw" before volatile load/store as it is
    // implemented in gcc. If memoperands is empty then assume that it aslo
    // maybe volatile load/store and insert "memw".
    if (MI.memoperands_empty() || (*MI.memoperands_begin())->isVolatile()) {
      BuildMI(*MBB, MI, DL, TII.get(Xtensa::MEMW));
    }
    return MBB;
  }
  default:
    llvm_unreachable("Unexpected instr type to insert");
  }
}
