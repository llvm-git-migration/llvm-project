//===- CallPromotionUtilsTest.cpp - CallPromotionUtils unit tests ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Analysis/CtxProfAnalysis.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/NoFolder.h"
#include "llvm/IR/PassInstrumentation.h"
#include "llvm/ProfileData/PGOCtxProfReader.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Testing/Support/SupportHelpers.h"
#include "gtest/gtest.h"

using namespace llvm;

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("UtilsTests", errs());
  return Mod;
}

class InlineFunctionTest : public testing::Test {
protected:
  LLVMContext C;
  std::unique_ptr<Module> M;
  llvm::unittest::TempFile ProfileFile;
  ModuleAnalysisManager MAM;

  const char *Profile = R"(
  [
    { "Guid": 1000,
      "Counters": [10, 2, 8],
      "Callsites": [
        [ { "Guid": 1001,
            "Counters": [2, 100],
            "Callsites": [[{"Guid": 1002, "Counters": [100]}]]}
        ],
        [ { "Guid": 1001,
            "Counters": [8, 500],
            "Callsites": [[{"Guid": 1002, "Counters": [500]}]]}
        ]
      ]
    }
  ]
  )";
  const char *IR = R"IR(
define i32 @entrypoint(i32 %x) !guid !0 {
  call void @llvm.instrprof.increment(ptr @entrypoint, i64 0, i32 3, i32 0)
  %t = icmp eq i32 %x, 0
  br i1 %t, label %yes, label %no
yes:
  call void @llvm.instrprof.increment(ptr @entrypoint, i64 0, i32 3, i32 1)
  call void @llvm.instrprof.callsite(ptr @entrypoint, i64 0, i32 2, i32 0, ptr @a)
  %call1 = call i32 @a(i32 %x)
  br label %exit
no:
  call void @llvm.instrprof.increment(ptr @entrypoint, i64 0, i32 3, i32 2)
  call void @llvm.instrprof.callsite(ptr @entrypoint, i64 0, i32 2, i32 1, ptr @a)
  %call2 = call i32 @a(i32 %x)
  br label %exit
exit:
  %ret = phi i32 [%call1, %yes], [%call2, %no]
  ret i32 %ret
}

define i32 @a(i32 %x) !guid !1 {
entry:
  call void @llvm.instrprof.increment(ptr @a, i64 0, i32 2, i32 0)
  br label %loop
loop:
  %indvar = phi i32 [%indvar.next, %loop], [0, %entry]
  call void @llvm.instrprof.increment(ptr @a, i64 0, i32 2, i32 1)
  %b = add i32 %x, %indvar
  call void @llvm.instrprof.callsite(ptr @a, i64 0, i32 1, i32 0, ptr @b)
  %inc = call i32 @b()
  %indvar.next = add i32 %indvar, %inc
  %cond = icmp slt i32 %indvar.next, %x
  br i1 %cond, label %loop, label %exit
exit:
  ret i32 8
}

define i32 @b() !guid !2 {
  call void @llvm.instrprof.increment(ptr @b, i64 0, i32 1, i32 0)
  ret i32 1
}

!0 = !{i64 1000}
!1 = !{i64 1001}
!2 = !{i64 1002}
)IR";

public:
  InlineFunctionTest() : ProfileFile("ctx_profile", "", "", /*Unique*/ true) {}

  void SetUp() override {
    M = parseIR(C, IR);
    ASSERT_TRUE(!!M);
    std::error_code EC;
    raw_fd_stream Out(ProfileFile.path(), EC);
    ASSERT_FALSE(EC);
    // "False" means no error.
    ASSERT_FALSE(llvm::createCtxProfFromJSON(Profile, Out));
    MAM.registerPass([&]() { return CtxProfAnalysis(ProfileFile.path()); });
    MAM.registerPass([&]() { return PassInstrumentationAnalysis(); });
  }
};

TEST_F(InlineFunctionTest, InlineWithCtxProf) {
  auto &CtxProf = MAM.getResult<CtxProfAnalysis>(*M);
  EXPECT_TRUE(!!CtxProf);
  auto *Caller = M->getFunction("entrypoint");
  CallBase *CB = [&]() -> CallBase * {
    for (auto &BB : *Caller)
      if (auto *Ins = CtxProfAnalysis::getBBInstrumentation(BB);
          Ins && Ins->getIndex()->getZExtValue() == 1)
        for (auto &I : BB)
          if (auto *CB = dyn_cast<CallBase>(&I);
              CB && CB->getCalledFunction() &&
              !CB->getCalledFunction()->isIntrinsic())
            return CB;
    return nullptr;
  }();
  ASSERT_NE(CB, nullptr);
  ASSERT_NE(CtxProfAnalysis::getCallsiteInstrumentation(*CB), nullptr);
  EXPECT_EQ(CtxProfAnalysis::getCallsiteInstrumentation(*CB)
                ->getIndex()
                ->getZExtValue(),
            0U);
  InlineFunctionInfo IFI;
  InlineResult IR = InlineFunction(*CB, IFI, CtxProf);
  EXPECT_TRUE(IR.isSuccess());
  std::string Str;
  raw_string_ostream OS(Str);
  CtxProfAnalysisPrinterPass Printer(
      OS, CtxProfAnalysisPrinterPass::PrintMode::JSON);
  Printer.run(*M, MAM);

  const char *Expected = R"(
  [
    { "Guid": 1000,
      "Counters": [10, 2, 8, 100],
      "Callsites": [
        [],
        [ { "Guid": 1001,
            "Counters": [8, 500],
            "Callsites": [[{"Guid": 1002, "Counters": [500]}]]}
        ],
        [{ "Guid": 1002, "Counters": [100]}]
      ]
    }
  ]
  )";

  auto ExpectedJSON = json::parse(Expected);
  ASSERT_TRUE(!!ExpectedJSON);
  auto ProducedJSON = json::parse(Str);
  ASSERT_TRUE(!!ProducedJSON);
  EXPECT_EQ(*ProducedJSON, *ExpectedJSON);
}