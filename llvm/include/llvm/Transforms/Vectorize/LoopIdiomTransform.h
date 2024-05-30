//===----------LoopIdiomTransform.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TRANSFORMS_VECTORIZE_LOOPIDIOMTRANSFORM_H
#define LLVM_LIB_TRANSFORMS_VECTORIZE_LOOPIDIOMTRANSFORM_H

#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"

namespace llvm {
enum class LoopIdiomTransformStyle { Masked, Predicated };

class LoopIdiomTransformPass : public PassInfoMixin<LoopIdiomTransformPass> {
  LoopIdiomTransformStyle VectorizeStyle = LoopIdiomTransformStyle::Masked;

  // The VF used in vectorizing the byte compare pattern.
  unsigned ByteCompareVF = 16;

public:
  LoopIdiomTransformPass() = default;
  explicit LoopIdiomTransformPass(LoopIdiomTransformStyle S)
      : VectorizeStyle(S) {}

  LoopIdiomTransformPass(LoopIdiomTransformStyle S, unsigned BCVF)
      : VectorizeStyle(S), ByteCompareVF(BCVF) {}

  PreservedAnalyses run(Loop &L, LoopAnalysisManager &AM,
                        LoopStandardAnalysisResults &AR, LPMUpdater &U);
};
} // namespace llvm
#endif // LLVM_LIB_TRANSFORMS_VECTORIZE_LOOPIDIOMTRANSFORM_H
