//=- DXILMetadataAnalysis.cpp - Representation of Module metadata -*- C++ -*=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/DXILMetadataAnalysis.h"
#include "llvm/ADT/APInt.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"

#define DEBUG_TYPE "dxil-metadata"

using namespace llvm;
using namespace dxil;

void ModuleMetadataInfo::dump(raw_ostream &OS) {
  OS << "Shader Model : " << Triple::getOSTypeName(ShaderModel) << "\n";
  OS << "DXIL Version : " << DXILVersion.getAsString() << "\n";
  OS << "Shader Stage : " << Triple::getEnvironmentTypeName(ShaderStage)
     << "\n";
}
//===----------------------------------------------------------------------===//
// DXILMetadataAnalysis and DXILMetadataAnalysisPrinterPass

// Provide an explicit template instantiation for the static ID.
AnalysisKey DXILMetadataAnalysis::Key;

llvm::dxil::ModuleMetadataInfo
DXILMetadataAnalysis::run(Module &M, ModuleAnalysisManager &AM) {
  ModuleMetadataInfo Data;
  return Data;
}

PreservedAnalyses
DXILMetadataAnalysisPrinterPass::run(Module &M, ModuleAnalysisManager &AM) {
  llvm::dxil::ModuleMetadataInfo &Data = AM.getResult<DXILMetadataAnalysis>(M);

  Data.dump(OS);
  return PreservedAnalyses::all();
}

//===----------------------------------------------------------------------===//
// DXILMetadataAnalysisWrapperPass

DXILMetadataAnalysisWrapperPass::DXILMetadataAnalysisWrapperPass()
    : ModulePass(ID) {
  initializeDXILMetadataAnalysisWrapperPassPass(
      *PassRegistry::getPassRegistry());
}

DXILMetadataAnalysisWrapperPass::~DXILMetadataAnalysisWrapperPass() = default;

void DXILMetadataAnalysisWrapperPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

bool DXILMetadataAnalysisWrapperPass::runOnModule(Module &M) {
  ModuleMetadata.reset(new llvm::dxil::ModuleMetadataInfo());
  Triple TT(Triple(M.getTargetTriple()));
  ModuleMetadata->DXILVersion = TT.getDXILVersion();
  ModuleMetadata->ShaderModel = TT.getOS();
  ModuleMetadata->ShaderStage = TT.getEnvironment();
  return false;
}

void DXILMetadataAnalysisWrapperPass::releaseMemory() {
  ModuleMetadata.reset();
}

void DXILMetadataAnalysisWrapperPass::print(raw_ostream &OS,
                                            const Module *) const {
  if (!ModuleMetadata) {
    OS << "No module metadata info has been built!\n";
    return;
  }
  ModuleMetadata->dump();
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD
void DXILMetadataAnalysisWrapperPass::dump() const { print(dbgs(), nullptr); }
#endif

INITIALIZE_PASS(DXILMetadataAnalysisWrapperPass, DEBUG_TYPE,
                "DXIL Module Metadata analysis", false, true)
char DXILMetadataAnalysisWrapperPass::ID = 0;
