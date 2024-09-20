//===- DependencyGraph.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/DependencyGraph.h"
#include "llvm/ADT/ArrayRef.h"

using namespace llvm::sandboxir;

// TODO: Move this to Utils once it lands.
/// \Returns the previous memory-dependency-candidate instruction before \p I in
/// the instruction stream.
static llvm::sandboxir::Instruction *
getPrevMemDepInst(llvm::sandboxir::Instruction *I) {
  for (I = I->getPrevNode(); I != nullptr; I = I->getPrevNode())
    if (I->isMemDepCandidate() || I->isStackSaveOrRestoreIntrinsic())
      return I;
  return nullptr;
}
/// \Returns the next memory-dependency-candidate instruction after \p I in the
/// instruction stream.
static llvm::sandboxir::Instruction *
getNextMemDepInst(llvm::sandboxir::Instruction *I) {
  for (I = I->getNextNode(); I != nullptr; I = I->getNextNode())
    if (I->isMemDepCandidate() || I->isStackSaveOrRestoreIntrinsic())
      return I;
  return nullptr;
}

DGNode *DGNode::getPrev(DependencyGraph &DAG) const {
  return DAG.getNodeOrNull(I->getPrevNode());
}
DGNode *DGNode::getNext(DependencyGraph &DAG) const {
  return DAG.getNodeOrNull(I->getNextNode());
}
DGNode *DGNode::getPrevMem(DependencyGraph &DAG) const {
  return DAG.getNodeOrNull(getPrevMemDepInst(I));
}
DGNode *DGNode::getNextMem(DependencyGraph &DAG) const {
  return DAG.getNodeOrNull(getNextMemDepInst(I));
}

#ifndef NDEBUG
void DGNode::print(raw_ostream &OS, bool PrintDeps) const {
  I->dumpOS(OS);
  if (PrintDeps) {
    OS << "\n";
    // Print memory preds.
    static constexpr const unsigned Indent = 4;
    for (auto *Pred : MemPreds) {
      OS.indent(Indent) << "<-";
      Pred->print(OS, false);
      OS << "\n";
    }
  }
}
void DGNode::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

DGNodeRange DGNodeRange::makeMemRange(DGNode *TopN, DGNode *BotN,
                                      DependencyGraph &DAG) {
  assert((TopN == BotN ||
          TopN->getInstruction()->comesBefore(BotN->getInstruction())) &&
         "Expected TopN before BotN!");
  // If TopN/BotN are not mem-dep candidate nodes we need to walk down/up the
  // chain and find the mem-dep ones.
  DGNode *MemTopN = TopN;
  DGNode *MemBotN = BotN;
  while (!MemTopN->isMem() && MemTopN != MemBotN)
    MemTopN = MemTopN->getNext(DAG);
  while (!MemBotN->isMem() && MemBotN != MemTopN)
    MemBotN = MemBotN->getPrev(DAG);
  // If we couldn't find a mem node in range TopN - BotN then it's empty.
  if (!MemTopN->isMem())
    return {};
  // Now that we have the mem-dep nodes, create and return the range.
  return DGNodeRange(MemDGNodeIterator(MemTopN, &DAG),
                     MemDGNodeIterator(MemBotN->getNextMem(DAG), &DAG));
}

#ifndef NDEBUG
void DGNodeRange::dump() const {
  for (const DGNode *N : *this)
    N->dump();
}
#endif // NDEBUG

InstrInterval DependencyGraph::extend(ArrayRef<Instruction *> Instrs) {
  if (Instrs.empty())
    return {};
  // TODO: For now create a chain of dependencies.
  InstrInterval Interval(Instrs);
  auto *TopI = Interval.top();
  auto *BotI = Interval.bottom();
  DGNode *LastN = getOrCreateNode(TopI);
  for (Instruction *I = TopI->getNextNode(), *E = BotI->getNextNode(); I != E;
       I = I->getNextNode()) {
    auto *N = getOrCreateNode(I);
    N->addMemPred(LastN);
    LastN = N;
  }
  return Interval;
}

#ifndef NDEBUG
void DependencyGraph::print(raw_ostream &OS) const {
  // InstrToNodeMap is unordered so we need to create an ordered vector.
  SmallVector<DGNode *> Nodes;
  Nodes.reserve(InstrToNodeMap.size());
  for (const auto &Pair : InstrToNodeMap)
    Nodes.push_back(Pair.second.get());
  // Sort them based on which one comes first in the BB.
  sort(Nodes, [](DGNode *N1, DGNode *N2) {
    return N1->getInstruction()->comesBefore(N2->getInstruction());
  });
  for (auto *N : Nodes)
    N->print(OS, /*PrintDeps=*/true);
}

void DependencyGraph::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG
