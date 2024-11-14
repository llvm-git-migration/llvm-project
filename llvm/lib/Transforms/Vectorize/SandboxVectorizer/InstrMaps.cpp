//===- InstructionMaps.cpp - Maps scalars to vectors and reverse ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/InstrMaps.h"
#include "llvm/Support/Debug.h"

namespace llvm::sandboxir {

void InstrMaps::clear() {
  OrigToVectorMap.clear();
  VectorToOrigLaneMap.clear();
}

#ifndef NDEBUG
void InstrMaps::print(raw_ostream &OS) const {
  for (auto &[Vec, Map] : VectorToOrigLaneMap) {
    OS << *Vec << "\n";
    SmallVector<std::pair<Value *, unsigned>> SortedOrigLanePairs;
    for (auto [Orig, Lane] : Map)
      SortedOrigLanePairs.push_back({Orig, Lane});
    sort(SortedOrigLanePairs, [](const auto &Pair1, const auto &Pair2) {
      int Lane1 = Pair1.second;
      int Lane2 = Pair2.second;
      return Lane1 < Lane2;
    });
    for (auto [Orig, Lane] : SortedOrigLanePairs)
      OS.indent(4) << "Lane " << Lane << " : " << *Orig << "\n";
  }
}

void InstrMaps::dump() const {
  print(dbgs());
  dbgs() << "\n";
}
#endif // NDEBUG

} // namespace llvm::sandboxir
