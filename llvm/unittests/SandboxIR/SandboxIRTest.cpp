//===- SandboxIRTest.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"

using namespace llvm;

struct SandboxIRTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("SandboxIRTest", errs());
  }
};

TEST_F(SandboxIRTest, UserInstantiation) {
  parseIR(C, R"IR(
define void @foo(i32 %v1) {
  ret void
}
)IR");
  Function &F = *M->getFunction("foo");
  auto *Ret = F.begin()->getTerminator();
  sandboxir::Context Ctx(C);
  [[maybe_unused]] sandboxir::User U(sandboxir::Value::ClassID::User, Ret, Ctx);
}

TEST_F(SandboxIRTest, FunctionArgumentConstantAndOpaqueInstInstantiation) {
  parseIR(C, R"IR(
define void @foo(i32 %v1) {
  %add = add i32 %v1, 42
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  llvm::BasicBlock *LLVMBB = &*LLVMF->begin();
  llvm::Instruction *LLVMAdd = &*LLVMBB->begin();
  auto *LLVMC = cast<llvm::Constant>(LLVMAdd->getOperand(1));
  sandboxir::Context Ctx(C);
  auto *LLVMArg0 = LLVMF->getArg(0);
  [[maybe_unused]] sandboxir::Function F(LLVMF, Ctx);
  [[maybe_unused]] sandboxir::Argument Arg0(LLVMArg0, Ctx);
  [[maybe_unused]] sandboxir::Constant Const0(LLVMC, Ctx);
  [[maybe_unused]] sandboxir::OpaqueInst Opaque(LLVMAdd, Ctx);
}
