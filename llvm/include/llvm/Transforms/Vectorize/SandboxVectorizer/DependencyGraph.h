//===- DependencyGraph.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the dependency graph used by the vectorizer's instruction
// scheduler.
//
// The nodes of the graph are objects of the `DGNode` class. Each `DGNode`
// object points to an instruction.
// The edges between `DGNode`s are implicitly defined by an ordered set of
// predecessor nodes, to save memory.
// Finally the whole dependency graph is an object of the `DependencyGraph`
// class, which also provides the API for creating/extending the graph from
// input Sandbox IR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_DEPENDENCYGRAPH_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_DEPENDENCYGRAPH_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/InstrInterval.h"

namespace llvm::sandboxir {

class DependencyGraph;

/// A DependencyGraph Node that points to an Instruction and contains memory
/// dependency edges.
class DGNode {
  Instruction *I;
  /// Memory predecessors.
  DenseSet<DGNode *> MemPreds;
  /// This is true if this may read/write memory, or if it has some ordering
  /// constraings, like with stacksave/stackrestore and alloca/inalloca.
  bool IsMem;

public:
  DGNode(Instruction *I) : I(I) {
    IsMem = I->isMemDepCandidate() ||
            (isa<AllocaInst>(I) && cast<AllocaInst>(I)->isUsedWithInAlloca()) ||
            I->isStackSaveOrRestoreIntrinsic();
  }
  DGNode(const DGNode &Other) = delete;
  Instruction *getInstruction() const { return I; }
  void addMemPred(DGNode *PredN) { MemPreds.insert(PredN); }
  /// \Returns all memory dependency predecessors.
  iterator_range<DenseSet<DGNode *>::const_iterator> memPreds() const {
    return make_range(MemPreds.begin(), MemPreds.end());
  }
  /// \Returns true if there is a memory dependency N->this.
  bool hasMemPred(DGNode *N) const { return MemPreds.count(N); }
  /// \Returns true if this may read/write memory, or if it has some ordering
  /// constraings, like with stacksave/stackrestore and alloca/inalloca.
  bool isMem() const { return IsMem; }
  /// \Returns the previous DGNode in program order.
  DGNode *getPrev(DependencyGraph &DAG) const;
  /// \Returns the next DGNode in program order.
  DGNode *getNext(DependencyGraph &DAG) const;
  /// Walks up the instruction chain looking for the next memory dependency
  /// candidate instruction.
  /// \Returns the corresponding DAG Node, or null if no instruction found.
  DGNode *getPrevMem(DependencyGraph &DAG) const;
  /// Walks down the instr chain looking for the next memory dependency
  /// candidate instruction.
  /// \Returns the corresponding DAG Node, or null if no instruction found.
  DGNode *getNextMem(DependencyGraph &DAG) const;

#ifndef NDEBUG
  void print(raw_ostream &OS, bool PrintDeps = true) const;
  friend raw_ostream &operator<<(DGNode &N, raw_ostream &OS) {
    N.print(OS);
    return OS;
  }
  LLVM_DUMP_METHOD void dump() const;
#endif // NDEBUG
};

/// Walks in the order of the instruction chain but skips non-mem Nodes.
/// This is used for building/updating the DAG.
class MemDGNodeIterator {
  DGNode *N;
  DependencyGraph *DAG;

public:
  using difference_type = std::ptrdiff_t;
  using value_type = DGNode;
  using pointer = value_type *;
  using reference = value_type &;
  using iterator_category = std::bidirectional_iterator_tag;
  MemDGNodeIterator(DGNode *N, DependencyGraph *DAG) : N(N), DAG(DAG) {
    assert((N == nullptr || N->isMem()) && "Expects mem node!");
  }
  MemDGNodeIterator &operator++() {
    assert(N != nullptr && "Already at end!");
    N = N->getNextMem(*DAG);
    return *this;
  }
  MemDGNodeIterator operator++(int) {
    auto ItCopy = *this;
    ++*this;
    return ItCopy;
  }
  MemDGNodeIterator &operator--() {
    N = N->getPrevMem(*DAG);
    return *this;
  }
  MemDGNodeIterator operator--(int) {
    auto ItCopy = *this;
    --*this;
    return ItCopy;
  }
  pointer operator*() { return N; }
  const DGNode *operator*() const { return N; }
  bool operator==(const MemDGNodeIterator &Other) const { return N == Other.N; }
  bool operator!=(const MemDGNodeIterator &Other) const {
    return !(*this == Other);
  }
};

/// A MemDGNodeIterator with convenience builders and dump().
class DGNodeRange : public iterator_range<MemDGNodeIterator> {
public:
  DGNodeRange(MemDGNodeIterator Begin, MemDGNodeIterator End)
      : iterator_range(Begin, End) {}
  /// An empty range.
  DGNodeRange()
      : iterator_range(MemDGNodeIterator(nullptr, nullptr),
                       MemDGNodeIterator(nullptr, nullptr)) {}
  /// Given \p TopN and \p BotN it finds their closest mem nodes in the range
  /// TopN to BotN and returns the corresponding mem range.
  /// Note: BotN (or its neighboring mem node) is included in the range.
  static DGNodeRange makeMemRange(DGNode *TopN, DGNode *BotN,
                                  DependencyGraph &DAG);
  static DGNodeRange makeEmptyMemRange() { return DGNodeRange(); }
#ifndef NDEBUG
  LLVM_DUMP_METHOD void dump() const;
#endif
};

class DependencyGraph {
private:
  DenseMap<Instruction *, std::unique_ptr<DGNode>> InstrToNodeMap;
  /// The DAG spans across all instructions in this interval.
  InstrInterval DAGInterval;

public:
  DependencyGraph() {}

  DGNode *getNode(Instruction *I) const {
    auto It = InstrToNodeMap.find(I);
    return It != InstrToNodeMap.end() ? It->second.get() : nullptr;
  }
  /// Like getNode() but returns nullptr if \p I is nullptr.
  DGNode *getNodeOrNull(Instruction *I) const {
    if (I == nullptr)
      return nullptr;
    return getNode(I);
  }
  DGNode *getOrCreateNode(Instruction *I) {
    auto [It, NotInMap] = InstrToNodeMap.try_emplace(I);
    if (NotInMap)
      It->second = std::make_unique<DGNode>(I);
    return It->second.get();
  }
  /// Build/extend the dependency graph such that it includes \p Instrs. Returns
  /// the interval spanning \p Instrs.
  InstrInterval extend(ArrayRef<Instruction *> Instrs);
#ifndef NDEBUG
  void print(raw_ostream &OS) const;
  LLVM_DUMP_METHOD void dump() const;
#endif // NDEBUG
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_DEPENDENCYGRAPH_H
