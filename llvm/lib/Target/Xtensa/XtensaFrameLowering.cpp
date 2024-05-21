//===- XtensaFrameLowering.cpp - Xtensa Frame Information -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Xtensa implementation of TargetFrameLowering class.
//
//===----------------------------------------------------------------------===//

#include "XtensaFrameLowering.h"
#include "XtensaInstrInfo.h"
#include "XtensaSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/IR/Function.h"

using namespace llvm;

XtensaFrameLowering::XtensaFrameLowering()
    : TargetFrameLowering(TargetFrameLowering::StackGrowsDown, Align(4), 0,
                          Align(4)) {}

bool XtensaFrameLowering::hasFP(const MachineFunction &MF) const {
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  return MF.getTarget().Options.DisableFramePointerElim(MF) ||
         MFI.hasVarSizedObjects();
}

void XtensaFrameLowering::emitPrologue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {
  assert(&MBB == &MF.front() && "Shrink-wrapping not yet implemented");
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const XtensaRegisterInfo *RegInfo = static_cast<const XtensaRegisterInfo *>(
      MF.getSubtarget().getRegisterInfo());
  const XtensaInstrInfo &TII =
      *static_cast<const XtensaInstrInfo *>(MF.getSubtarget().getInstrInfo());
  MachineBasicBlock::iterator MBBI = MBB.begin();
  DebugLoc DL = MBBI != MBB.end() ? MBBI->getDebugLoc() : DebugLoc();
  unsigned SP = Xtensa::SP;
  unsigned FP = RegInfo->getFrameRegister(MF);
  MachineModuleInfo &MMI = MF.getMMI();
  const MCRegisterInfo *MRI = MMI.getContext().getRegisterInfo();

  // First, compute final stack size.
  uint64_t StackSize = MFI.getStackSize();
  uint64_t PrevStackSize = StackSize;

  // Round up StackSize to 16*N
  StackSize += (16 - StackSize) & 0xf;

  // No need to allocate space on the stack.
  if (StackSize == 0 && !MFI.adjustsStack())
    return;

  // Adjust stack.
  TII.adjustStackPtr(SP, -StackSize, MBB, MBBI);

  // emit ".cfi_def_cfa_offset StackSize"
  unsigned CFIIndex =
      MF.addFrameInst(MCCFIInstruction::cfiDefCfaOffset(nullptr, StackSize));
  BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
      .addCFIIndex(CFIIndex);

  const std::vector<CalleeSavedInfo> &CSI = MFI.getCalleeSavedInfo();

  if (CSI.size()) {
    // Find the instruction past the last instruction that saves a
    // callee-saved register to the stack.
    for (unsigned i = 0; i < CSI.size(); ++i)
      ++MBBI;

    // Iterate over list of callee-saved registers and emit .cfi_offset
    // directives.
    for (const auto &I : CSI) {
      int64_t Offset = MFI.getObjectOffset(I.getFrameIdx());
      Register Reg = I.getReg();

      unsigned CFIIndex = MF.addFrameInst(MCCFIInstruction::createOffset(
          nullptr, MRI->getDwarfRegNum(Reg, 1), Offset));
      BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
          .addCFIIndex(CFIIndex);
    }
  }

  // if framepointer enabled, set it to point to the stack pointer.
  if (hasFP(MF)) {
    // Insert instruction "move $fp, $sp" at this location.
    BuildMI(MBB, MBBI, DL, TII.get(Xtensa::OR), FP)
        .addReg(SP)
        .addReg(SP)
        .setMIFlag(MachineInstr::FrameSetup);

    // emit ".cfi_def_cfa_register $fp"
    unsigned CFIIndex = MF.addFrameInst(MCCFIInstruction::createDefCfaRegister(
        nullptr, MRI->getDwarfRegNum(FP, true)));
    BuildMI(MBB, MBBI, DL, TII.get(TargetOpcode::CFI_INSTRUCTION))
        .addCFIIndex(CFIIndex);
  }

  if (StackSize != PrevStackSize) {
    MFI.setStackSize(StackSize);

    for (int i = MFI.getObjectIndexBegin(); i < MFI.getObjectIndexEnd(); i++) {
      if (!MFI.isDeadObjectIndex(i)) {
        int64_t SPOffset = MFI.getObjectOffset(i);

        if (SPOffset < 0)
          MFI.setObjectOffset(i, SPOffset - StackSize + PrevStackSize);
      }
    }
  }
}

void XtensaFrameLowering::emitEpilogue(MachineFunction &MF,
                                       MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator MBBI = MBB.getLastNonDebugInstr();
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const XtensaRegisterInfo *RegInfo = static_cast<const XtensaRegisterInfo *>(
      MF.getSubtarget().getRegisterInfo());
  const XtensaInstrInfo &TII =
      *static_cast<const XtensaInstrInfo *>(MF.getSubtarget().getInstrInfo());
  DebugLoc DL = MBBI->getDebugLoc();
  unsigned SP = Xtensa::SP;
  unsigned FP = RegInfo->getFrameRegister(MF);

  // if framepointer enabled, restore the stack pointer.
  if (hasFP(MF)) {
    // Find the first instruction that restores a callee-saved register.
    MachineBasicBlock::iterator I = MBBI;

    for (unsigned i = 0; i < MFI.getCalleeSavedInfo().size(); ++i)
      --I;

    BuildMI(MBB, I, DL, TII.get(Xtensa::OR), SP).addReg(FP).addReg(FP);
  }

  // Get the number of bytes from FrameInfo
  uint64_t StackSize = MFI.getStackSize();

  if (!StackSize)
    return;

  // Adjust stack.
  TII.adjustStackPtr(SP, StackSize, MBB, MBBI);
}

bool XtensaFrameLowering::spillCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    ArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  MachineFunction *MF = MBB.getParent();

  MachineBasicBlock &EntryBlock = *(MF->begin());
  const TargetInstrInfo &TII = *MF->getSubtarget().getInstrInfo();

  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    // Add the callee-saved register as live-in. Do not add if the register is
    // A0 and return address is taken, because it will be implemented in
    // method XtensaTargetLowering::LowerRETURNADDR.
    // It's killed at the spill, unless the register is RA and return address
    // is taken.
    Register Reg = CSI[i].getReg();
    bool IsA0AndRetAddrIsTaken =
        (Reg == Xtensa::A0) && MF->getFrameInfo().isReturnAddressTaken();
    if (!IsA0AndRetAddrIsTaken)
      EntryBlock.addLiveIn(Reg);

    // Insert the spill to the stack frame.
    bool IsKill = !IsA0AndRetAddrIsTaken;
    const TargetRegisterClass *RC = TRI->getMinimalPhysRegClass(Reg);
    TII.storeRegToStackSlot(EntryBlock, MI, Reg, IsKill, CSI[i].getFrameIdx(),
                            RC, TRI, Register());
  }

  return true;
}

bool XtensaFrameLowering::restoreCalleeSavedRegisters(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
    MutableArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const {
  return TargetFrameLowering::restoreCalleeSavedRegisters(MBB, MI, CSI, TRI);
}

// Eliminate ADJCALLSTACKDOWN, ADJCALLSTACKUP pseudo instructions
MachineBasicBlock::iterator XtensaFrameLowering::eliminateCallFramePseudoInstr(
    MachineFunction &MF, MachineBasicBlock &MBB,
    MachineBasicBlock::iterator I) const {
  const XtensaInstrInfo &TII =
      *static_cast<const XtensaInstrInfo *>(MF.getSubtarget().getInstrInfo());

  if (!hasReservedCallFrame(MF)) {
    int64_t Amount = I->getOperand(0).getImm();

    if (I->getOpcode() == Xtensa::ADJCALLSTACKDOWN)
      Amount = -Amount;

    unsigned SP = Xtensa::SP;
    TII.adjustStackPtr(SP, Amount, MBB, I);
  }

  return MBB.erase(I);
}

void XtensaFrameLowering::determineCalleeSaves(MachineFunction &MF,
                                               BitVector &SavedRegs,
                                               RegScavenger *RS) const {
  MachineFrameInfo &MFI = MF.getFrameInfo();
  const XtensaRegisterInfo *RegInfo = static_cast<const XtensaRegisterInfo *>(
      MF.getSubtarget().getRegisterInfo());
  unsigned FP = RegInfo->getFrameRegister(MF);

  TargetFrameLowering::determineCalleeSaves(MF, SavedRegs, RS);

  // Mark $fp as used if function has dedicated frame pointer.
  if (hasFP(MF))
    SavedRegs.set(FP);

  // Set scavenging frame index if necessary.
  uint64_t MaxSPOffset = MFI.estimateStackSize(MF);

  if (isInt<12>(MaxSPOffset))
    return;

  const TargetRegisterClass &RC = Xtensa::ARRegClass;
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();
  unsigned Size = TRI->getSpillSize(RC);
  Align Alignment = TRI->getSpillAlign(RC);
  int FI = MF.getFrameInfo().CreateStackObject(Size, Alignment, false);
  RS->addScavengingFrameIndex(FI);
}
