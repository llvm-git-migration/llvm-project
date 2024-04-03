//===- RISCVOptWInstrs.cpp - MI W instruction optimizations ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This pass attempts to reassociate expressions like
//   (sh3add Z, (add X, (slli Y, 6)))
// To
//   (sh3add (sh3add Y, Z), X)
//
// This reduces number of instructions needing by spreading the shift amount
// across to 2 sh3adds. This can be generalized to sh1add/sh2add and other
// shift amounts.
//
// TODO: We can also support slli.uw by using shXadd.uw or the inner shXadd.

#include "RISCV.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-opt-zba"
#define RISCV_OPT_ZBA_NAME "RISC-V Optimize Zba"

namespace {

class RISCVOptZba : public MachineFunctionPass {
public:
  static char ID;

  RISCVOptZba() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_OPT_ZBA_NAME; }
};

} // end anonymous namespace

char RISCVOptZba::ID = 0;
INITIALIZE_PASS(RISCVOptZba, DEBUG_TYPE, RISCV_OPT_ZBA_NAME, false, false)

FunctionPass *llvm::createRISCVOptZbaPass() { return new RISCVOptZba(); }

static bool findShift(Register Reg, const MachineBasicBlock &MBB,
                      MachineInstr *&Shift, const MachineRegisterInfo &MRI) {
  if (!Reg.isVirtual())
    return false;

  Shift = MRI.getVRegDef(Reg);
  if (!Shift || Shift->getOpcode() != RISCV::SLLI ||
      Shift->getParent() != &MBB || !MRI.hasOneNonDBGUse(Reg))
    return false;

  return true;
}

bool RISCVOptZba::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  MachineRegisterInfo &MRI = MF.getRegInfo();
  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  const RISCVInstrInfo &TII = *ST.getInstrInfo();

  if (!ST.hasStdExtZba())
    return false;

  bool MadeChange = true;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
      unsigned OuterShiftAmt;
      switch (MI.getOpcode()) {
      default:
        continue;
      case RISCV::SH1ADD: OuterShiftAmt = 1; break;
      case RISCV::SH2ADD: OuterShiftAmt = 2; break;
      case RISCV::SH3ADD: OuterShiftAmt = 3; break;
      }

      // Second operand must be virtual.
      Register UnshiftedReg = MI.getOperand(2).getReg();
      if (!UnshiftedReg.isVirtual())
        continue;

      MachineInstr *Add = MRI.getVRegDef(UnshiftedReg);
      if (!Add || Add->getOpcode() != RISCV::ADD || Add->getParent() != &MBB ||
          !MRI.hasOneNonDBGUse(UnshiftedReg))
        continue;

      Register AddReg0 = Add->getOperand(1).getReg();
      Register AddReg1 = Add->getOperand(2).getReg();

      MachineInstr *InnerShift;
      Register X;
      if (findShift(AddReg0, MBB, InnerShift, MRI))
        X = AddReg1;
      else if (findShift(AddReg1, MBB, InnerShift, MRI))
        X = AddReg0;
      else
        continue;

      unsigned InnerShiftAmt = InnerShift->getOperand(2).getImm();

      // The inner shift amount must be at least as large as the outer shift
      // amount.
      if (OuterShiftAmt > InnerShiftAmt)
        continue;

      unsigned InnerOpc;
      switch (InnerShiftAmt - OuterShiftAmt) {
      default:
        continue;
      case 0: InnerOpc = RISCV::ADD;    break;
      case 1: InnerOpc = RISCV::SH1ADD; break;
      case 2: InnerOpc = RISCV::SH2ADD; break;
      case 3: InnerOpc = RISCV::SH3ADD; break;
      }

      Register Y = InnerShift->getOperand(1).getReg();
      Register Z = MI.getOperand(1).getReg();

      Register NewReg = MRI.createVirtualRegister(&RISCV::GPRRegClass);
      BuildMI(MBB, MI, MI.getDebugLoc(), TII.get(InnerOpc), NewReg)
          .addReg(Y)
          .addReg(Z);
      BuildMI(MBB, MI, MI.getDebugLoc(), TII.get(MI.getOpcode()),
              MI.getOperand(0).getReg())
          .addReg(NewReg)
          .addReg(X);

      MI.eraseFromParent();
      Add->eraseFromParent();
      InnerShift->eraseFromParent();
      MadeChange = true;
    }
  }

  return MadeChange;
}
