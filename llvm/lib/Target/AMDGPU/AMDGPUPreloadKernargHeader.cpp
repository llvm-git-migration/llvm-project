//===- AMDGPUPreloadKernargHeader.cpp - Preload Kernarg Header ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass handles the creation of the backwards compatability layer
/// for kernarg prealoding. Code may be compiled with the feature enabled, while
/// the kernel is executed on hardware without firmware support.
///
/// To avoid the need for recompilation, we insert a block at the beginning of
/// the kernel that is responsible for loading the kernel arguments into SGPRs
/// using s_load instructions which setup the registers exactly as they would be
/// by firmware if the code were executed on a system that supported kernarg
/// preladoing.
///
/// This essentially allows for two entry points for the kernel. Firmware that
/// supports the feature will automatically jump past the first 256 bytes of the
/// program, skipping the backwards compatibility layer and directly beginning
/// execution on the fast code path.
///
/// This pass should be run as late as possible, to avoid any optimization that
/// may assume that padding is dead code or that the prologue added here is a
/// true predecessor of the kernel entry block.
//===----------------------------------------------------------------------===//

#include "AMDGPUPreloadKernargHeader.h"
#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/TargetParser/TargetParser.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-preload-kernarg-header"

namespace {

struct LoadConfig {
  unsigned Size;
  const TargetRegisterClass *RegClass;
  unsigned Opcode;
  Register LoadReg;

  // Constructor for the static config array
  LoadConfig(unsigned S, const TargetRegisterClass *RC, unsigned Op)
      : Size(S), RegClass(RC), Opcode(Op), LoadReg(AMDGPU::NoRegister) {}

  // Constructor for the return value
  LoadConfig(unsigned S, const TargetRegisterClass *RC, unsigned Op,
             Register Reg)
      : Size(S), RegClass(RC), Opcode(Op), LoadReg(Reg) {}
};

class AMDGPUPreloadKernargHeader {
public:
  AMDGPUPreloadKernargHeader(MachineFunction &MF);

  bool run();

private:
  MachineFunction &MF;
  const GCNSubtarget &ST;
  const SIMachineFunctionInfo &MFI;
  const SIInstrInfo &TII;
  const TargetRegisterInfo &TRI;

  // Create a new block before the entry point to the kernel. Firmware that
  // supports preloading kernel arguments will automatically jump past this
  // block to the alternative kernel entry point.
  void createBackCompatBlock();

  // Add instructions to load kernel arguments into SGPRs, returns the number of
  // s_load instructions added.
  unsigned addBackCompatLoads(MachineBasicBlock *BackCompatMBB,
                              Register KernargSegmentPtr,
                              unsigned NumKernargPreloadSGPRs);
};

class AMDGPUPreloadKernargHeaderLegacy : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUPreloadKernargHeaderLegacy() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AMDGPU Preload Kernarg Header";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // end anonymous namespace

char AMDGPUPreloadKernargHeaderLegacy::ID = 0;

INITIALIZE_PASS(AMDGPUPreloadKernargHeaderLegacy, DEBUG_TYPE,
                "AMDGPU Preload Kernarg Header", false, false)

char &llvm::AMDGPUPreloadKernargHeaderLegacyID =
    AMDGPUPreloadKernargHeaderLegacy::ID;

FunctionPass *llvm::createAMDGPUPreloadKernargHeaderLegacyPass() {
  return new AMDGPUPreloadKernargHeaderLegacy();
}

bool AMDGPUPreloadKernargHeaderLegacy::runOnMachineFunction(
    MachineFunction &MF) {
  return AMDGPUPreloadKernargHeader(MF).run();
}

AMDGPUPreloadKernargHeader::AMDGPUPreloadKernargHeader(MachineFunction &MF)
    : MF(MF), ST(MF.getSubtarget<GCNSubtarget>()),
      MFI(*MF.getInfo<SIMachineFunctionInfo>()), TII(*ST.getInstrInfo()),
      TRI(*ST.getRegisterInfo()) {}

bool AMDGPUPreloadKernargHeader::run() {
  if (!ST.hasKernargPreload())
    return false;

  unsigned NumPreloadSGPRs = MFI.getNumKernargPreloadedSGPRs();
  if (NumPreloadSGPRs <= 0)
    return false;

  if (MF.begin() == MF.end())
    return false;

  createBackCompatBlock();

  return true;
}

void AMDGPUPreloadKernargHeader::createBackCompatBlock() {
  auto KernelEntryMBB = MF.begin();
  MachineBasicBlock *BackCompatMBB = MF.CreateMachineBasicBlock();
  MF.insert(KernelEntryMBB, BackCompatMBB);
  BackCompatMBB->addSuccessor(&*KernelEntryMBB);

  assert(MFI.getUserSGPRInfo().hasKernargSegmentPtr());
  Register KernargSegmentPtr = MFI.getArgInfo().KernargSegmentPtr.getRegister();
  BackCompatMBB->addLiveIn(KernargSegmentPtr);

  unsigned NumKernargPreloadSGPRs = MFI.getNumKernargPreloadedSGPRs();
  unsigned NumInstrs = 0;

  // Load kernel arguments to SGPRs
  NumInstrs += addBackCompatLoads(BackCompatMBB, KernargSegmentPtr,
                                  NumKernargPreloadSGPRs);

  AMDGPU::IsaVersion IV = AMDGPU::getIsaVersion(ST.getCPU());
  unsigned Waitcnt =
      AMDGPU::encodeWaitcnt(IV, getVmcntBitMask(IV), getExpcntBitMask(IV), 0);

  // Wait for loads to complete
  BuildMI(BackCompatMBB, DebugLoc(), TII.get(AMDGPU::S_WAITCNT))
      .addImm(Waitcnt);
  NumInstrs++;

  // Set PC to the actual kernel entry point.  Add padding to fill out the rest
  // of the backcompat block. The total number of bytes must be 256.
  for (unsigned I = 0; I < 64 - NumInstrs; ++I) {
    BuildMI(BackCompatMBB, DebugLoc(), TII.get(AMDGPU::S_BRANCH))
        .addMBB(&*KernelEntryMBB);
  }
}

// Find the largest possible load size that fits with SGRP alignment
static LoadConfig getLoadParameters(const TargetRegisterInfo &TRI,
                                    Register KernargPreloadSGPR,
                                    unsigned NumKernargPreloadSGPRs) {
  static const LoadConfig Configs[] = {
      {8, &AMDGPU::SReg_256RegClass, AMDGPU::S_LOAD_DWORDX8_IMM},
      {4, &AMDGPU::SReg_128RegClass, AMDGPU::S_LOAD_DWORDX4_IMM},
      {2, &AMDGPU::SReg_64RegClass, AMDGPU::S_LOAD_DWORDX2_IMM},
      {1, &AMDGPU::SReg_32RegClass, AMDGPU::S_LOAD_DWORD_IMM}};

  // Find the largest possible load size
  for (const auto &Config : Configs) {
    if (NumKernargPreloadSGPRs >= Config.Size) {
      Register LoadReg = TRI.getMatchingSuperReg(KernargPreloadSGPR,
                                                 AMDGPU::sub0, Config.RegClass);
      if (LoadReg != AMDGPU::NoRegister)
        return LoadConfig(Config.Size, Config.RegClass, Config.Opcode, LoadReg);
    }
  }

  // Fallback to a single register
  return LoadConfig(1, &AMDGPU::SReg_32RegClass, AMDGPU::S_LOAD_DWORD_IMM,
                    KernargPreloadSGPR);
}

unsigned AMDGPUPreloadKernargHeader::addBackCompatLoads(
    MachineBasicBlock *BackCompatMBB, Register KernargSegmentPtr,
    unsigned NumKernargPreloadSGPRs) {
  Register KernargPreloadSGPR = MFI.getArgInfo().FirstKernArgPreloadReg;
  unsigned Offset = 0;
  unsigned NumLoads = 0;

  // Fill all user SGPRs used for kernarg preloading with sequential data from
  // the kernarg segment
  while (NumKernargPreloadSGPRs > 0) {
    LoadConfig Config =
        getLoadParameters(TRI, KernargPreloadSGPR, NumKernargPreloadSGPRs);

    BuildMI(BackCompatMBB, DebugLoc(), TII.get(Config.Opcode), Config.LoadReg)
        .addReg(KernargSegmentPtr)
        .addImm(Offset)
        .addImm(0);

    Offset += 4 * Config.Size;
    KernargPreloadSGPR = KernargPreloadSGPR.asMCReg() + Config.Size;
    NumKernargPreloadSGPRs -= Config.Size;
    NumLoads++;
  }

  return NumLoads;
}

PreservedAnalyses
AMDGPUPreloadKernargHeaderPass::run(MachineFunction &MF,
                                    MachineFunctionAnalysisManager &) {
  if (!AMDGPUPreloadKernargHeader(MF).run())
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
