#include <llvm/CodeGen/MachineBasicBlock.h>
#include <llvm/CodeGen/MachineBranchProbabilityInfo.h>
#include <llvm/CodeGen/MachineDominators.h>
#include <llvm/CodeGen/MachineFunctionPass.h>
#include <llvm/CodeGen/MachineLoopInfo.h>
#include <llvm/CodeGen/SSAIfConv.h>
#include <llvm/CodeGen/TargetInstrInfo.h>
#include <llvm/CodeGen/TargetRegisterInfo.h>
#include <llvm/CodeGen/TargetSchedule.h>
#include <llvm/CodeGen/TargetSubtargetInfo.h>
#include <llvm/InitializePasses.h>

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"

using namespace llvm;

namespace {
#define DEBUG_TYPE "amdgpu-if-cvt"
const char PassName[] = "AMDGPU if conversion";

class AMDGPUIfConverter : public MachineFunctionPass {
  const SIInstrInfo *TII = nullptr;
  TargetSchedModel SchedModel;
  MachineDominatorTree *DomTree = nullptr;
  MachineBranchProbabilityInfo *MBPI = nullptr;
  MachineLoopInfo *Loops = nullptr;

  static constexpr unsigned BlockInstrLimit = 30;
  static constexpr bool Stress = false;
  SSAIfConv IfConv{DEBUG_TYPE, BlockInstrLimit, Stress};

public:
  static char ID;

  AMDGPUIfConverter() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool tryConvertIf(MachineBasicBlock *);
  bool shouldConvertIf();

  StringRef getPassName() const override { return PassName; }
};

char AMDGPUIfConverter::ID = 0;

void AMDGPUIfConverter::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineBranchProbabilityInfoWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  AU.addRequired<MachineLoopInfoWrapperPass>();
  AU.addPreserved<MachineLoopInfoWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool AMDGPUIfConverter::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  const auto &STI = MF.getSubtarget<GCNSubtarget>();
  if (!STI.hasGFX10_3Insts())
    return false;

  TII = STI.getInstrInfo();
  SchedModel.init(&STI);
  DomTree = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  Loops = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
  MBPI = &getAnalysis<MachineBranchProbabilityInfoWrapperPass>().getMBPI();

  bool Changed = false;
  IfConv.runOnMachineFunction(MF);

  for (auto *DomNode : post_order(DomTree))
    if (tryConvertIf(DomNode->getBlock()))
      Changed = true;

  return Changed;
}

unsigned getReversedVCMPXOpcode(unsigned Opcode) {
  // TODO: this is a placeholder for the real function
  switch (Opcode) {
  case AMDGPU::V_CMPX_LT_I32_nosdst_e64:
    return AMDGPU::V_CMPX_GE_I32_nosdst_e64;
  default:
    errs() << "unhandled: " << Opcode << "\n";
    llvm_unreachable("unhandled vcmp opcode");
  }
}

bool needsPredication(const SIInstrInfo *TII, const MachineInstr &I) {
  return TII->isVALU(I) || TII->isVMEM(I);
}

struct ExecPredicate : ifcvt::PredicationStrategy {
  const SIInstrInfo *TII;
  const SIRegisterInfo *RegInfo;

  MachineInstr *Cmp = nullptr;

  ExecPredicate(const SIInstrInfo *TII)
      : TII(TII), RegInfo(&TII->getRegisterInfo()) {}

  bool canConvertIf(MachineBasicBlock *Head, MachineBasicBlock *TBB,
                    MachineBasicBlock *FBB, MachineBasicBlock *Tail,
                    ArrayRef<MachineOperand> Cond) override {

    // check that the cmp is just before the branch and that it is promotable to
    // v_cmpx
    const unsigned SupportedBranchOpc[]{
        AMDGPU::S_CBRANCH_SCC0, AMDGPU::S_CBRANCH_SCC1, AMDGPU::S_CBRANCH_VCCNZ,
        AMDGPU::S_CBRANCH_VCCZ};

    MachineInstr &CBranch = *Head->getFirstInstrTerminator();
    if (!llvm::is_contained(SupportedBranchOpc, CBranch.getOpcode()))
      return false;

    auto CmpInstr = std::next(CBranch.getReverseIterator());
    if (CmpInstr == Head->instr_rend())
      return false;

    Register SCCorVCC = Cond[1].getReg();
    bool ModifiesConditionReg = CmpInstr->modifiesRegister(SCCorVCC, RegInfo);
    if (!ModifiesConditionReg)
      return false;

    Cmp = &*CmpInstr;

    unsigned CmpOpc = Cmp->getOpcode();
    if (TII->isSALU(*Cmp))
      CmpOpc = TII->getVALUOp(*Cmp);
    if (AMDGPU::getVCMPXOpFromVCMP(CmpOpc) == -1) {
      errs() << *Cmp << "\n";
      return false;
    }

    auto NeedsPredication = [&](const MachineInstr &I) {
      return needsPredication(TII, I);
    };
    auto BlockNeedsPredication = [&](const MachineBasicBlock *MBB) {
      if (MBB == Tail)
        return false;
      auto Insts = llvm::make_range(MBB->begin(), MBB->getFirstTerminator());
      return llvm::any_of(Insts, NeedsPredication);
    };

    MachineBasicBlock *Blocks[] = {TBB, FBB};

    if (llvm::none_of(Blocks, BlockNeedsPredication))
      return false;

    return true;
  }

  bool canPredicate(const MachineInstr &I) override {

    // TODO: relax this condition, if exec is masked, check that it goes back to
    // normal
    // TODO: what about scc or vcc ? Are they taken into acount in the MBB
    // live-ins ?
    MCRegister Exec = RegInfo->getExec();
    bool ModifiesExec = I.modifiesRegister(Exec, RegInfo);
    if (ModifiesExec)
      return false;

    if (needsPredication(TII, I))
      return true;

    bool DontMoveAcrossStore = true;
    bool IsSpeculatable = I.isDereferenceableInvariantLoad() ||
                          I.isSafeToMove(DontMoveAcrossStore);
    if (IsSpeculatable)
      return true;

    return false;
  }

  bool predicateBlock(MachineBasicBlock *MBB, ArrayRef<MachineOperand> Cond,
                      bool Reverse) override {
    // save exec
    MachineFunction &MF = *MBB->getParent();
    SIMachineFunctionInfo *MFI = MF.getInfo<SIMachineFunctionInfo>();

    Register ExecBackup = MFI->getSGPRForEXECCopy();

    const DebugLoc &CmpLoc = Cmp->getDebugLoc();

    auto FirstInstruction = MBB->begin();
    const bool IsSCCLive =
        false; // asume not since the live-ins are supposed to be empty
    TII->insertScratchExecCopy(MF, *MBB, FirstInstruction, CmpLoc, ExecBackup,
                               IsSCCLive);

    // mask exec
    unsigned CmpOpc = Cmp->getOpcode();
    if (TII->isSALU(*Cmp))
      CmpOpc = TII->getVALUOp(*Cmp);

    CmpOpc = AMDGPU::getVCMPXOpFromVCMP(CmpOpc);
    if (Reverse)
      CmpOpc = getReversedVCMPXOpcode(CmpOpc);

    // TODO: handle this properly. The second block may kill those registers.
    Cmp->getOperand(0).setIsKill(false);
    Cmp->getOperand(1).setIsKill(false);

    auto VCmpX = BuildMI(*MBB, FirstInstruction, CmpLoc, TII->get(CmpOpc));
    VCmpX->addOperand(Cmp->getOperand(0));
    VCmpX->addOperand(Cmp->getOperand(1));

    // restore exec
    TII->restoreExec(MF, *MBB, MBB->end(), DebugLoc(), ExecBackup);

    return true;
  }

  ~ExecPredicate() override = default;
};

/// Update the dominator tree after if-conversion erased some blocks.
void updateDomTree(MachineDominatorTree *DomTree, const SSAIfConv &IfConv,
                   ArrayRef<MachineBasicBlock *> Removed) {
  // convertIf can remove TBB, FBB, and Tail can be merged into Head.
  // TBB and FBB should not dominate any blocks.
  // Tail children should be transferred to Head.
  MachineDomTreeNode *HeadNode = DomTree->getNode(IfConv.Head);
  for (auto *B : Removed) {
    MachineDomTreeNode *Node = DomTree->getNode(B);
    assert(Node != HeadNode && "Cannot erase the head node");
    while (Node->getNumChildren()) {
      assert(Node->getBlock() == IfConv.Tail && "Unexpected children");
      DomTree->changeImmediateDominator(Node->back(), HeadNode);
    }
    DomTree->eraseNode(B);
  }
}

/// Update LoopInfo after if-conversion.
void updateLoops(MachineLoopInfo *Loops,
                 ArrayRef<MachineBasicBlock *> Removed) {
  // If-conversion doesn't change loop structure, and it doesn't mess with back
  // edges, so updating LoopInfo is simply removing the dead blocks.
  for (auto *B : Removed)
    Loops->removeBlock(B);
}

bool AMDGPUIfConverter::shouldConvertIf() {
  // TODO: cost model
  return true;
}

bool AMDGPUIfConverter::tryConvertIf(MachineBasicBlock *MBB) {
  ExecPredicate Predicate{TII};
  bool Changed = false;
  while (IfConv.canConvertIf(MBB, Predicate) && shouldConvertIf()) {
    // If-convert MBB and update analyses.
    SmallVector<MachineBasicBlock *, 4> RemoveBlocks;
    IfConv.convertIf(RemoveBlocks, Predicate);
    Changed = true;
    updateDomTree(DomTree, IfConv, RemoveBlocks);
    for (MachineBasicBlock *MBB : RemoveBlocks)
      MBB->eraseFromParent();
    updateLoops(Loops, RemoveBlocks);
  }
  return Changed;
}

} // namespace

char &llvm::AMDGPUIfConverterID = AMDGPUIfConverter::ID;
INITIALIZE_PASS_BEGIN(AMDGPUIfConverter, DEBUG_TYPE, PassName, false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfoWrapperPass)
INITIALIZE_PASS_END(AMDGPUIfConverter, DEBUG_TYPE, PassName, false, false)