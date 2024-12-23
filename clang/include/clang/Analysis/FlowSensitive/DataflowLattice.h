//===- DataflowLattice.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines base types for building lattices to be used in dataflow
//  analyses that run over Control-Flow Graphs (CFGs).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DATAFLOWLATTICE_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DATAFLOWLATTICE_H

#include "llvm/Support/Casting.h"
#include "llvm/Support/ExtensibleRTTI.h"
#include <memory>

namespace clang {
namespace dataflow {

/// Effect indicating whether a lattice operation resulted in a new value.
enum class LatticeEffect {
  Unchanged,
  Changed,
};
// DEPRECATED. Use `LatticeEffect`.
using LatticeJoinEffect = LatticeEffect;

class DataflowLattice
    : public llvm::RTTIExtends<DataflowLattice, llvm::RTTIRoot> {
public:
  inline static char ID = 0;
  DataflowLattice() = default;

  /// Joins two type-erased lattice elements by computing their least upper
  /// bound.
  virtual LatticeEffect join(const DataflowLattice &) = 0;

  virtual std::unique_ptr<DataflowLattice> clone() = 0;

  /// Chooses a lattice element that approximates the current element at a
  /// program point, given the previous element at that point. Places the
  /// widened result in the current element (`Current`). Widening is optional --
  /// it is only needed to either accelerate convergence (for lattices with
  /// non-trivial height) or guarantee convergence (for lattices with infinite
  /// height).
  ///
  /// Returns an indication of whether any changes were made to `Current` in
  /// order to widen. This saves a separate call to `isEqualTypeErased` after
  /// the widening.
  virtual LatticeEffect widen(const DataflowLattice &Previous) {
    return isEqual(Previous) ? LatticeEffect::Unchanged
                             : LatticeEffect::Changed;
  }

  /// Returns true if and only if the two given type-erased lattice elements are
  /// equal.
  virtual bool isEqual(const DataflowLattice &) const = 0;
};

using DataflowLatticePtr = std::unique_ptr<DataflowLattice>;

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_DATAFLOWLATTICE_H
