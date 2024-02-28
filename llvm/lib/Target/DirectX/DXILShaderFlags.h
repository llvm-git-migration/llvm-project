//===- DXILShaderFlags.h - DXIL Shader Flags helper objects ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file This file contains helper objects and APIs for working with DXIL
///       Shader Flags.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_DIRECTX_DXILSHADERFLAGS_H
#define LLVM_TARGET_DIRECTX_DXILSHADERFLAGS_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

namespace llvm {
class Module;
class GlobalVariable;

namespace dxil {

struct ComputedShaderFlags {
#define DXIL_MODULE_FLAG(bit, featureBit, FlagName, Str) bool FlagName : 1;
#include "llvm/Support/DXILConstants.def"

#define DXIL_MODULE_FLAG(bit, featureBit, FlagName, Str) FlagName = false;
  ComputedShaderFlags() {
#include "llvm/Support/DXILConstants.def"
  }
  static uint64_t getFeatureInfoMask(int featureBit) {
    return featureBit != -1 ? (uint64_t)1 << featureBit : 0ull;
  }
  uint64_t getFeatureInfo() const {
    uint64_t FeatureInfo = 0;
#define DXIL_MODULE_FLAG(bit, featureBit, FlagName, Str)                       \
  FeatureInfo |= FlagName ? getFeatureInfoMask(featureBit) : 0ull;

#include "llvm/Support/DXILConstants.def"

    return FeatureInfo;
  }

  operator uint64_t() const {
    uint64_t FlagValue = 0;
#define DXIL_MODULE_FLAG(bit, featureBit, FlagName, Str)                       \
  FlagValue |= FlagName ? (uint64_t)1 << bit : 0ull;
#include "llvm/Support/DXILConstants.def"
    return FlagValue;
  }

  static ComputedShaderFlags computeFlags(Module &M);
  void print(raw_ostream &OS = dbgs()) const;
  LLVM_DUMP_METHOD void dump() const { print(); }
};

class ShaderFlagsAnalysis : public AnalysisInfoMixin<ShaderFlagsAnalysis> {
  friend AnalysisInfoMixin<ShaderFlagsAnalysis>;
  static AnalysisKey Key;

public:
  ShaderFlagsAnalysis() = default;

  using Result = ComputedShaderFlags;

  ComputedShaderFlags run(Module &M, ModuleAnalysisManager &AM);
};

/// Printer pass for ShaderFlagsAnalysis results.
class ShaderFlagsAnalysisPrinter
    : public PassInfoMixin<ShaderFlagsAnalysisPrinter> {
  raw_ostream &OS;

public:
  explicit ShaderFlagsAnalysisPrinter(raw_ostream &OS) : OS(OS) {}
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

/// Wrapper pass for the legacy pass manager.
///
/// This is required because the passes that will depend on this are codegen
/// passes which run through the legacy pass manager.
class ShaderFlagsAnalysisWrapper : public ModulePass {
  ComputedShaderFlags Flags;

public:
  static char ID;

  ShaderFlagsAnalysisWrapper() : ModulePass(ID) {}

  const ComputedShaderFlags &getShaderFlags() { return Flags; }

  bool runOnModule(Module &M) override {
    Flags = ComputedShaderFlags::computeFlags(M);
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }
};

} // namespace dxil
} // namespace llvm

#endif // LLVM_TARGET_DIRECTX_DXILSHADERFLAGS_H
