//===--- InstallAPI/Context.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/InstallAPI/Context.h"
#include "clang/AST/ASTContext.h"
#include "llvm/TextAPI/TextAPIWriter.h"

using namespace clang;
using namespace clang::installapi;
using namespace llvm::MachO;

void InstallAPIConsumer::HandleTranslationUnit(ASTContext &Context) {
  if (Context.getDiagnostics().hasErrorOccurred())
    return;
}

std::unique_ptr<ASTConsumer>
InstallAPIAction::CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
  InstallAPIContext Ctx;
  return std::make_unique<InstallAPIConsumer>(std::move(Ctx));
}
