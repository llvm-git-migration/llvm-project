//===- LoopUtils.h -  LoopUtils Support ---------------------*- C++
//-*-=============//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions for the action framework. This framework
// allows for external entities to control certain actions taken by the compiler
// by registering handler functions.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_IR_LOOP_UTILS_H
#define MLIR_IR_LOOP_UTILS_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir {

// Gathers all maximal sub-blocks of operations that do not themselves
// include a `OpTy` (an operation could have a descendant `OpTy` though
// in its tree). Ignore the block terminators.
template <typename OpTy>
struct JamBlockGatherer {
  // Store iterators to the first and last op of each sub-block found.
  llvm::SmallVector<std::pair<Block::iterator, Block::iterator>> subBlocks;

  // This is a linear time walk.
  void walk(Operation *op) {
    for (auto &region : op->getRegions())
      for (auto &block : region)
        walk(block);
  }

  void walk(Block &block) {
    assert(!block.empty() && block.back().hasTrait<OpTrait::IsTerminator>() &&
           "expected block to have a terminator");
    for (auto it = block.begin(), e = std::prev(block.end()); it != e;) {
      auto subBlockStart = it;
      while (it != e && !isa<OpTy>(&*it))
        ++it;
      if (it != subBlockStart)
        subBlocks.emplace_back(subBlockStart, std::prev(it));
      // Process all for ops that appear next.
      while (it != e && isa<OpTy>(&*it))
        walk(&*it++);
    }
  }
};

} // namespace mlir

#endif // MLIR_IR_LOOP_UTILS_H
