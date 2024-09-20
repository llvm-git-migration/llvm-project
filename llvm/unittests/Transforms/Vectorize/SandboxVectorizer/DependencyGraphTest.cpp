//===- DependencyGraphTest.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/DependencyGraph.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/SandboxIR/SandboxIR.h"
#include "llvm/Support/SourceMgr.h"
#include "gmock/gmock-matchers.h"
#include "gtest/gtest.h"

using namespace llvm;

struct DependencyGraphTest : public testing::Test {
  LLVMContext C;
  std::unique_ptr<Module> M;

  void parseIR(LLVMContext &C, const char *IR) {
    SMDiagnostic Err;
    M = parseAssemblyString(IR, Err, C);
    if (!M)
      Err.print("DependencyGraphTest", errs());
  }
};

TEST_F(DependencyGraphTest, DGNode_IsMem) {
  parseIR(C, R"IR(
declare void @llvm.sideeffect()
declare void @llvm.pseudoprobe(i64, i64, i32, i64)
declare void @llvm.fake.use(...)
declare void @bar()
define void @foo(i8 %v1, ptr %ptr) {
  store i8 %v1, ptr %ptr
  %ld0 = load i8, ptr %ptr
  %add = add i8 %v1, %v1
  %stacksave = call ptr @llvm.stacksave()
  call void @llvm.stackrestore(ptr %stacksave)
  call void @llvm.sideeffect()
  call void @llvm.pseudoprobe(i64 42, i64 1, i32 0, i64 -1)
  call void @llvm.fake.use(ptr %ptr)
  call void @bar()
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *Store = cast<sandboxir::StoreInst>(&*It++);
  auto *Load = cast<sandboxir::LoadInst>(&*It++);
  auto *Add = cast<sandboxir::BinaryOperator>(&*It++);
  auto *StackSave = cast<sandboxir::CallInst>(&*It++);
  auto *StackRestore = cast<sandboxir::CallInst>(&*It++);
  auto *SideEffect = cast<sandboxir::CallInst>(&*It++);
  auto *PseudoProbe = cast<sandboxir::CallInst>(&*It++);
  auto *FakeUse = cast<sandboxir::CallInst>(&*It++);
  auto *Call = cast<sandboxir::CallInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  sandboxir::DependencyGraph DAG;
  DAG.extend({&*BB->begin(), BB->getTerminator()});
  EXPECT_TRUE(DAG.getNode(Store)->isMem());
  EXPECT_TRUE(DAG.getNode(Load)->isMem());
  EXPECT_FALSE(DAG.getNode(Add)->isMem());
  EXPECT_TRUE(DAG.getNode(StackSave)->isMem());
  EXPECT_TRUE(DAG.getNode(StackRestore)->isMem());
  EXPECT_FALSE(DAG.getNode(SideEffect)->isMem());
  EXPECT_FALSE(DAG.getNode(PseudoProbe)->isMem());
  EXPECT_TRUE(DAG.getNode(FakeUse)->isMem());
  EXPECT_TRUE(DAG.getNode(Call)->isMem());
  EXPECT_FALSE(DAG.getNode(Ret)->isMem());
}

TEST_F(DependencyGraphTest, Basic) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v0, i8 %v1) {
  store i8 %v0, ptr %ptr
  store i8 %v1, ptr %ptr
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *S0 = cast<sandboxir::StoreInst>(&*It++);
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);
  sandboxir::DependencyGraph DAG;
  auto Span = DAG.extend({&*BB->begin(), BB->getTerminator()});
  // Check extend().
  EXPECT_EQ(Span.top(), &*BB->begin());
  EXPECT_EQ(Span.bottom(), BB->getTerminator());

  sandboxir::DGNode *N0 = DAG.getNode(S0);
  sandboxir::DGNode *N1 = DAG.getNode(S1);
  sandboxir::DGNode *N2 = DAG.getNode(Ret);
  // Check getInstruction().
  EXPECT_EQ(N0->getInstruction(), S0);
  EXPECT_EQ(N1->getInstruction(), S1);
  // Check hasMemPred()
  EXPECT_TRUE(N1->hasMemPred(N0));
  EXPECT_FALSE(N0->hasMemPred(N1));

  // Check memPreds().
  EXPECT_TRUE(N0->memPreds().empty());
  EXPECT_THAT(N1->memPreds(), testing::ElementsAre(N0));
  EXPECT_THAT(N2->memPreds(), testing::ElementsAre(N1));
}

TEST_F(DependencyGraphTest, DGNode_getPrev_getNext_getPrevMem_getNextMem) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v0, i8 %v1) {
  store i8 %v0, ptr %ptr
  add i8 %v0, %v0
  store i8 %v1, ptr %ptr
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *S0 = cast<sandboxir::StoreInst>(&*It++);
  auto *Add = cast<sandboxir::BinaryOperator>(&*It++);
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  sandboxir::DependencyGraph DAG;
  DAG.extend({&*BB->begin(), BB->getTerminator()});

  sandboxir::DGNode *S0N = DAG.getNode(S0);
  sandboxir::DGNode *AddN = DAG.getNode(Add);
  sandboxir::DGNode *S1N = DAG.getNode(S1);
  sandboxir::DGNode *RetN = DAG.getNode(Ret);

  EXPECT_EQ(S0N->getPrev(DAG), nullptr);
  EXPECT_EQ(S0N->getNext(DAG), AddN);
  EXPECT_EQ(S0N->getPrevMem(DAG), nullptr);
  EXPECT_EQ(S0N->getNextMem(DAG), S1N);

  EXPECT_EQ(AddN->getPrev(DAG), S0N);
  EXPECT_EQ(AddN->getNext(DAG), S1N);
  EXPECT_EQ(AddN->getPrevMem(DAG), S0N);
  EXPECT_EQ(AddN->getNextMem(DAG), S1N);

  EXPECT_EQ(S1N->getPrev(DAG), AddN);
  EXPECT_EQ(S1N->getNext(DAG), RetN);
  EXPECT_EQ(S1N->getPrevMem(DAG), S0N);
  EXPECT_EQ(S1N->getNextMem(DAG), nullptr);

  EXPECT_EQ(RetN->getPrev(DAG), S1N);
  EXPECT_EQ(RetN->getNext(DAG), nullptr);
  EXPECT_EQ(RetN->getPrevMem(DAG), S1N);
  EXPECT_EQ(RetN->getNextMem(DAG), nullptr);
}

TEST_F(DependencyGraphTest, DGNodeRange) {
  parseIR(C, R"IR(
define void @foo(ptr %ptr, i8 %v0, i8 %v1) {
  add i8 %v0, %v0
  store i8 %v0, ptr %ptr
  add i8 %v0, %v0
  store i8 %v1, ptr %ptr
  ret void
}
)IR");
  llvm::Function *LLVMF = &*M->getFunction("foo");
  sandboxir::Context Ctx(C);
  auto *F = Ctx.createFunction(LLVMF);
  auto *BB = &*F->begin();
  auto It = BB->begin();
  auto *Add0 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *S0 = cast<sandboxir::StoreInst>(&*It++);
  auto *Add1 = cast<sandboxir::BinaryOperator>(&*It++);
  auto *S1 = cast<sandboxir::StoreInst>(&*It++);
  auto *Ret = cast<sandboxir::ReturnInst>(&*It++);

  sandboxir::DependencyGraph DAG;
  DAG.extend({&*BB->begin(), BB->getTerminator()});

  sandboxir::DGNode *Add0N = DAG.getNode(Add0);
  sandboxir::DGNode *S0N = DAG.getNode(S0);
  sandboxir::DGNode *Add1N = DAG.getNode(Add1);
  sandboxir::DGNode *S1N = DAG.getNode(S1);
  sandboxir::DGNode *RetN = DAG.getNode(Ret);

  // Check empty range.
  EXPECT_THAT(sandboxir::DGNodeRange::makeEmptyMemRange(),
              testing::ElementsAre());

  // Both TopN and BotN are memory.
  EXPECT_THAT(sandboxir::DGNodeRange::makeMemRange(S0N, S1N, DAG),
              testing::ElementsAre(S0N, S1N));
  // Only TopN is memory.
  EXPECT_THAT(sandboxir::DGNodeRange::makeMemRange(S0N, RetN, DAG),
              testing::ElementsAre(S0N, S1N));
  EXPECT_THAT(sandboxir::DGNodeRange::makeMemRange(S0N, Add1N, DAG),
              testing::ElementsAre(S0N));
  // Only BotN is memory.
  EXPECT_THAT(sandboxir::DGNodeRange::makeMemRange(Add0N, S1N, DAG),
              testing::ElementsAre(S0N, S1N));
  EXPECT_THAT(sandboxir::DGNodeRange::makeMemRange(Add0N, S0N, DAG),
              testing::ElementsAre(S0N));
  // Neither TopN or BotN is memory.
  EXPECT_THAT(sandboxir::DGNodeRange::makeMemRange(Add0N, RetN, DAG),
              testing::ElementsAre(S0N, S1N));
  EXPECT_THAT(sandboxir::DGNodeRange::makeMemRange(Add0N, Add0N, DAG),
              testing::ElementsAre());
}
