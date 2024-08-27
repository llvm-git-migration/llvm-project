#ifndef LLVM_SSA_IF_CONV_H
#define LLVM_SSA_IF_CONV_H

#include "llvm/ADT/SparseSet.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

//===----------------------------------------------------------------------===//
//                                 SSAIfConv
//===----------------------------------------------------------------------===//
//
// The SSAIfConv class performs if-conversion on SSA form machine code after
// determining if it is possible. The class contains no heuristics; external
// code should be used to determine when if-conversion is a good idea.
//
// SSAIfConv can convert both triangles and diamonds:
//
//   Triangle: Head              Diamond: Head
//              | \                       /  \_
//              |  \                     /    |
//              |  [TF]BB              FBB    TBB
//              |  /                     \    /
//              | /                       \  /
//             Tail                       Tail
//
// Instructions in the conditional blocks TBB and/or FBB are spliced into the
// Head block, and phis in the Tail block are converted to select instructions.
//
namespace llvm {
class SSAIfConv;

namespace ifcvt {
struct PredicationStrategy {
  virtual bool canConvertIf(MachineBasicBlock *Head, MachineBasicBlock *TBB,
                            MachineBasicBlock *FBB, MachineBasicBlock *Tail,
                            ArrayRef<MachineOperand> Cond) {
    return true;
  }
  virtual bool canPredicate(const MachineInstr &I) = 0;
  virtual bool predicateBlock(MachineBasicBlock *Succ,
                              ArrayRef<MachineOperand> Cond, bool Reverse) = 0;
  virtual ~PredicationStrategy() = default;
};
} // namespace ifcvt

class SSAIfConv {
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  MachineRegisterInfo *MRI;

  // TODO INITIALIZE
  const char *DEBUG_TYPE;
  unsigned BlockInstrLimit;
  bool Stress;

  struct Statistics {
    unsigned NumTrianglesSeen = 0;
    unsigned NumDiamondsSeen = 0;
    unsigned NumTrianglesConv = 0;
    unsigned NumDiamondsConv = 0;
  };

  Statistics S;

public:
  SSAIfConv(const char *DEBUG_TYPE, unsigned BlockInstrLimit, bool Stress)
      : DEBUG_TYPE(DEBUG_TYPE), BlockInstrLimit(BlockInstrLimit),
        Stress(Stress) {}

  template <typename CounterTy>
  void updateStatistics(CounterTy &NumDiamondsSeen, CounterTy &NumDiamondsConv,
                        CounterTy &NumTrianglesSeen,
                        CounterTy &NumTrianglesConv) const {
    NumDiamondsSeen += S.NumDiamondsSeen;
    NumDiamondsConv += S.NumDiamondsConv;
    NumTrianglesSeen += S.NumTrianglesSeen;
    NumTrianglesConv += S.NumTrianglesConv;
  }

  /// The block containing the conditional branch.
  MachineBasicBlock *Head;

  /// The block containing phis after the if-then-else.
  MachineBasicBlock *Tail;

  /// The 'true' conditional block as determined by analyzeBranch.
  MachineBasicBlock *TBB;

  /// The 'false' conditional block as determined by analyzeBranch.
  MachineBasicBlock *FBB;

  /// isTriangle - When there is no 'else' block, either TBB or FBB will be
  /// equal to Tail.
  bool isTriangle() const { return TBB == Tail || FBB == Tail; }

  /// Returns the Tail predecessor for the True side.
  MachineBasicBlock *getTPred() const { return TBB == Tail ? Head : TBB; }

  /// Returns the Tail predecessor for the  False side.
  MachineBasicBlock *getFPred() const { return FBB == Tail ? Head : FBB; }

  /// Information about each phi in the Tail block.
  struct PHIInfo {
    MachineInstr *PHI;
    unsigned TReg = 0, FReg = 0;
    // Latencies from Cond+Branch, TReg, and FReg to DstReg.
    int CondCycles = 0, TCycles = 0, FCycles = 0;

    PHIInfo(MachineInstr *phi) : PHI(phi) {}
  };

  SmallVector<PHIInfo, 8> PHIs;

  /// The branch condition determined by analyzeBranch.
  SmallVector<MachineOperand, 4> Cond;

private:
  /// Instructions in Head that define values used by the conditional blocks.
  /// The hoisted instructions must be inserted after these instructions.
  SmallPtrSet<MachineInstr *, 8> InsertAfter;

  /// Register units clobbered by the conditional blocks.
  BitVector ClobberedRegUnits;

  // Scratch pad for findInsertionPoint.
  SparseSet<unsigned> LiveRegUnits;

  /// Insertion point in Head for speculatively executed instructions form TBB
  /// and FBB.
  MachineBasicBlock::iterator InsertionPoint;

  /// Return true if all non-terminator instructions in MBB can be safely
  /// predicated.
  bool canPredicateInstrs(MachineBasicBlock *MBB,
                          ifcvt::PredicationStrategy &Predicate);

  /// Scan through instruction dependencies and update InsertAfter array.
  /// Return false if any dependency is incompatible with if conversion.
  bool InstrDependenciesAllowIfConv(MachineInstr *I);

  /// Predicate all instructions of the basic block with current condition
  /// except for terminators. Reverse the condition if ReversePredicate is set.
  void PredicateBlock(MachineBasicBlock *MBB, bool ReversePredicate);

  /// Find a valid insertion point in Head.
  bool findInsertionPoint();

  /// Replace PHI instructions in Tail with selects.
  void replacePHIInstrs();

  /// Insert selects and rewrite PHI operands to use them.
  void rewritePHIOperands();

public:
  /// runOnMachineFunction - Initialize per-function data structures.
  void runOnMachineFunction(MachineFunction &MF);

  /// canConvertIf - If the sub-CFG headed by MBB can be if-converted,
  /// initialize the internal state, and return true.
  /// If predicate is set try to predicate the block otherwise try to
  /// speculatively execute it.
  bool canConvertIf(MachineBasicBlock *MBB, ifcvt::PredicationStrategy &S);

  /// convertIf - If-convert the last block passed to canConvertIf(), assuming
  /// it is possible. Add any blocks that are to be erased to RemoveBlocks.
  void convertIf(SmallVectorImpl<MachineBasicBlock *> &RemoveBlocks,
                 ifcvt::PredicationStrategy &S);
};
} // namespace llvm

#endif
