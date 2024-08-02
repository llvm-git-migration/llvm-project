//=- DXILMetadataAnalysis.h - Representation of Module metadata --*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DXILMETADATA_H
#define LLVM_ANALYSIS_DXILMETADATA_H

#include "llvm/ADT/MapVector.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Pass.h"
#include "llvm/Support/DXILABI.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {

namespace dxil {

struct ModuleMetadataInfo {
  VersionTuple DXILVersion;
  Triple::OSType ShaderModel;
  Triple::EnvironmentType ShaderStage;

  void dump(raw_ostream &OS = errs());
};

} // namespace dxil

class DXILMetadataAnalysis : public AnalysisInfoMixin<DXILMetadataAnalysis> {
  friend AnalysisInfoMixin<DXILMetadataAnalysis>;

  static AnalysisKey Key;

public:
  using Result = dxil::ModuleMetadataInfo;
  /// Gather module metadata info for the module \c M.
  dxil::ModuleMetadataInfo run(Module &M, ModuleAnalysisManager &AM);
};

/// Printer pass for the \c DXILMetadataAnalysis results.
class DXILMetadataAnalysisPrinterPass
    : public PassInfoMixin<DXILMetadataAnalysisPrinterPass> {
  raw_ostream &OS;

public:
  explicit DXILMetadataAnalysisPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);

  static bool isRequired() { return true; }
};

/// Wrapper pass to be used by other passes using legacy pass manager
class DXILMetadataAnalysisWrapperPass : public ModulePass {
  std::unique_ptr<dxil::ModuleMetadataInfo> ModuleMetadata;

public:
  static char ID; // Class identification, replacement for typeinfo

  DXILMetadataAnalysisWrapperPass();
  ~DXILMetadataAnalysisWrapperPass() override;

  const dxil::ModuleMetadataInfo &getModuleMetadata() const {
    return *ModuleMetadata;
  }
  dxil::ModuleMetadataInfo &getModuleMetadata() { return *ModuleMetadata; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnModule(Module &M) override;
  void releaseMemory() override;

  void print(raw_ostream &OS, const Module *M) const override;
  void dump() const;
};

} // namespace llvm

#endif // LLVM_ANALYSIS_DXILMETADATA_H
