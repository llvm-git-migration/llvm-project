//===-- WebAssemblyCleanCodeAfterTrap.cpp - Argument instruction moving ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file moves ARGUMENT instructions after ScheduleDAG scheduling.
///
/// Arguments are really live-in registers, however, since we use virtual
/// registers and LLVM doesn't support live-in virtual registers, we're
/// currently making do with ARGUMENT instructions which are placed at the top
/// of the entry block. The trick is to get them to *stay* at the top of the
/// entry block.
///
/// The ARGUMENTS physical register keeps these instructions pinned in place
/// during liveness-aware CodeGen passes, however one thing which does not
/// respect this is the ScheduleDAG scheduler. This pass is therefore run
/// immediately after that.
///
/// This is all hopefully a temporary solution until we find a better solution
/// for describing the live-in nature of arguments.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "WebAssemblyUtilities.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-clean-code-after-trap"

namespace {
class WebAssemblyCleanCodeAfterTrap final : public MachineFunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblyCleanCodeAfterTrap() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "WebAssembly Clean Unreachable Code After Trap"; }

  bool runOnMachineFunction(MachineFunction &MF) override;
};
} // end anonymous namespace

char WebAssemblyCleanCodeAfterTrap::ID = 0;
INITIALIZE_PASS(WebAssemblyCleanCodeAfterTrap, DEBUG_TYPE,
                "Clean unreachable code after trap instruction", false, false)

FunctionPass *llvm::createWebAssemblyCleanCodeAfterTrap() {
  return new WebAssemblyCleanCodeAfterTrap();
}

bool WebAssemblyCleanCodeAfterTrap::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG({
    dbgs() << "********** CleanUnreachableCodeAfterTrap **********\n"
           << "********** Function: " << MF.getName() << '\n';
  });

  bool Changed = false;

  for (MachineBasicBlock & BB : MF) {
    bool HasTerminator = false;
    llvm::SmallVector<MachineInstr*> RemoveMI{};
    for (MachineInstr & MI : BB) {
      if (HasTerminator)
        RemoveMI.push_back(&MI);
      if (MI.hasProperty(MCID::Trap) && MI.isTerminator())
        HasTerminator = true;
    }
    if (!RemoveMI.empty()) {
      Changed = true;
      LLVM_DEBUG({
        for (MachineInstr *MI : RemoveMI) {
          llvm::dbgs() << "* remove ";
          MI->print(llvm::dbgs());
        }
      });
      for (MachineInstr * MI : RemoveMI)
        MI->eraseFromParent();
    }
  }
  return Changed;
}
