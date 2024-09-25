//===- unittests/IR/DroppedVariableStatsTest.cpp - TimePassesHandler tests
//----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/SourceMgr.h"
#include "gtest/gtest.h"
#include <gtest/gtest.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassInstrumentation.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/PassTimingInfo.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;
namespace llvm {
void initializePassTest1Pass(PassRegistry &);

static std::unique_ptr<Module> parseIR(LLVMContext &C, const char *IR) {
  SMDiagnostic Err;
  std::unique_ptr<Module> Mod = parseAssemblyString(IR, Err, C);
  if (!Mod)
    Err.print("AbstractCallSiteTests", errs());
  return Mod;
}
} // namespace llvm

namespace {

// This test ensures that if a #dbg_value and an instruction that exists in the
// same scope as that #dbg_value are both deleted as a result of an optimization
// pass, debug information is considered not dropped.
TEST(DroppedVariableStats, BothDeleted) {
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *IR =
      "; Function Attrs: mustprogress nounwind ssp uwtable(sync)\n"
      "define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 "
      "noundef %x) local_unnamed_addr #0 !dbg !9 {\n"
      "entry:\n"
      "#dbg_value(i32 %x, !15, !DIExpression(), !16)\n"
      "%add = add nsw i32 %x, 1, !dbg !17\n"
      "ret i32 0\n"
      "}\n"
      "!llvm.dbg.cu = !{!0}\n"
      "!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}\n"
      "!llvm.ident = !{!8}\n"
      "!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: "
      "!1, producer: \"clang version 20.0.0git "
      "(git@github.com:llvm/llvm-project.git "
      "baff49d3a0ef8e0848a726656ebf6e7b310e5113)\", isOptimized: true, "
      "runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, "
      "nameTableKind: Apple, sysroot: \"/\")\n"
      "!1 = !DIFile(filename: \"/tmp/code.cpp\", directory: "
      "\"/Users/shubham/Development/llvm-project/build_ninja\", checksumkind: "
      "CSK_MD5, checksum: \"719364c4b07176af8515cac6bd21008c\")\n"
      "!2 = !{i32 7, !\"Dwarf Version\", i32 5}\n"
      "!3 = !{i32 2, !\"Debug Info Version\", i32 3}\n"
      "!4 = !{i32 1, !\"wchar_size\", i32 4}\n"
      "!5 = !{i32 8, !\"PIC Level\", i32 2}\n"
      "!6 = !{i32 7, !\"uwtable\", i32 1}\n"
      "!7 = !{i32 7, !\"frame-pointer\", i32 1}\n"
      "!8 = !{!\"clang version 20.0.0git (git@github.com:llvm/llvm-project.git "
      "baff49d3a0ef8e0848a726656ebf6e7b310e5113)\"}\n"
      "!9 = distinct !DISubprogram(name: \"foo\", linkageName: \"_Z3fooi\", "
      "scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, flags: "
      "DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition "
      "| DISPFlagOptimized, unit: !0, retainedNodes: !14)\n"
      "!10 = !DIFile(filename: \"/tmp/code.cpp\", directory: \"\", "
      "checksumkind: CSK_MD5, checksum: \"719364c4b07176af8515cac6bd21008c\")\n"
      "!11 = !DISubroutineType(types: !12)\n"
      "!12 = !{!13, !13}\n"
      "!13 = !DIBasicType(name: \"int\", size: 32, encoding: DW_ATE_signed)\n"
      "!14 = !{!15}\n"
      "!15 = !DILocalVariable(name: \"x\", arg: 1, scope: !9, file: !10, line: "
      "1, type: !13)\n"
      "!16 = !DILocation(line: 0, scope: !9)\n"
      "!17 = !DILocation(line: 2, column: 11, scope: !9)\n";

  std::unique_ptr<llvm::Module> M = parseIR(C, IR);
  ASSERT_TRUE(M);

  DroppedVariableStats Stats(true);
  Stats.runBeforePass("Test",
                      llvm::Any(const_cast<const llvm::Module *>(M.get())));

  // Remove instructions
  for (auto &F : *M.get()) {
    for (auto &I : instructions(&F)) {
      I.dropDbgRecords();
      I.eraseFromParent();
      break;
    }
    break;
  }
  PreservedAnalyses PA;
  Stats.runAfterPass("Test",
                     llvm::Any(const_cast<const llvm::Module *>(M.get())), PA);
  ASSERT_EQ(Stats.getPassDroppedVariables(), false);
}

} // end anonymous namespace

namespace {

// This test ensures that if a #dbg_value is dropped after an optimization pass,
// but an instruction that shares the same scope as the #dbg_value still exists,
// debug information is conisdered dropped.
TEST(DroppedVariableStats, DbgValLost) {
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *IR =
      "; Function Attrs: mustprogress nounwind ssp uwtable(sync)\n"
      "define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 "
      "noundef %x) local_unnamed_addr #0 !dbg !9 {\n"
      "entry:\n"
      "#dbg_value(i32 %x, !15, !DIExpression(), !16)\n"
      "%add = add nsw i32 %x, 1, !dbg !17\n"
      "ret i32 0\n"
      "}\n"
      "!llvm.dbg.cu = !{!0}\n"
      "!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}\n"
      "!llvm.ident = !{!8}\n"
      "!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: "
      "!1, producer: \"clang version 20.0.0git "
      "(git@github.com:llvm/llvm-project.git "
      "baff49d3a0ef8e0848a726656ebf6e7b310e5113)\", isOptimized: true, "
      "runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, "
      "nameTableKind: Apple, sysroot: \"/\")\n"
      "!1 = !DIFile(filename: \"/tmp/code.cpp\", directory: "
      "\"/Users/shubham/Development/llvm-project/build_ninja\", checksumkind: "
      "CSK_MD5, checksum: \"719364c4b07176af8515cac6bd21008c\")\n"
      "!2 = !{i32 7, !\"Dwarf Version\", i32 5}\n"
      "!3 = !{i32 2, !\"Debug Info Version\", i32 3}\n"
      "!4 = !{i32 1, !\"wchar_size\", i32 4}\n"
      "!5 = !{i32 8, !\"PIC Level\", i32 2}\n"
      "!6 = !{i32 7, !\"uwtable\", i32 1}\n"
      "!7 = !{i32 7, !\"frame-pointer\", i32 1}\n"
      "!8 = !{!\"clang version 20.0.0git (git@github.com:llvm/llvm-project.git "
      "baff49d3a0ef8e0848a726656ebf6e7b310e5113)\"}\n"
      "!9 = distinct !DISubprogram(name: \"foo\", linkageName: \"_Z3fooi\", "
      "scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, flags: "
      "DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition "
      "| DISPFlagOptimized, unit: !0, retainedNodes: !14)\n"
      "!10 = !DIFile(filename: \"/tmp/code.cpp\", directory: \"\", "
      "checksumkind: CSK_MD5, checksum: \"719364c4b07176af8515cac6bd21008c\")\n"
      "!11 = !DISubroutineType(types: !12)\n"
      "!12 = !{!13, !13}\n"
      "!13 = !DIBasicType(name: \"int\", size: 32, encoding: DW_ATE_signed)\n"
      "!14 = !{!15}\n"
      "!15 = !DILocalVariable(name: \"x\", arg: 1, scope: !9, file: !10, line: "
      "1, type: !13)\n"
      "!16 = !DILocation(line: 0, scope: !9)\n"
      "!17 = !DILocation(line: 2, column: 11, scope: !9)\n";

  std::unique_ptr<llvm::Module> M = parseIR(C, IR);
  ASSERT_TRUE(M);

  DroppedVariableStats Stats(true);
  Stats.runBeforePass("Test",
                      llvm::Any(const_cast<const llvm::Module *>(M.get())));

  // Remove instructions
  for (auto &F : *M.get()) {
    for (auto &I : instructions(&F)) {
      I.dropDbgRecords();
      break;
    }
    break;
  }
  PreservedAnalyses PA;
  Stats.runAfterPass("Test",
                     llvm::Any(const_cast<const llvm::Module *>(M.get())), PA);
  ASSERT_EQ(Stats.getPassDroppedVariables(), true);
}

// This test ensures that if a #dbg_value is dropped after an optimization pass,
// but an instruction that has an unrelated scope as the #dbg_value still
// exists, debug information is conisdered not dropped.
TEST(DroppedVariableStats, UnrelatedScopes) {
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *IR =
      "; Function Attrs: mustprogress nounwind ssp uwtable(sync)\n"
      "define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 "
      "noundef %x) local_unnamed_addr #0 !dbg !9 {\n"
      "entry:\n"
      "#dbg_value(i32 %x, !15, !DIExpression(), !16)\n"
      "%add = add nsw i32 %x, 1, !dbg !17\n"
      "ret i32 0\n"
      "}\n"
      "!llvm.dbg.cu = !{!0}\n"
      "!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}\n"
      "!llvm.ident = !{!8}\n"
      "!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: "
      "!1, producer: \"clang version 20.0.0git "
      "(git@github.com:llvm/llvm-project.git "
      "baff49d3a0ef8e0848a726656ebf6e7b310e5113)\", isOptimized: true, "
      "runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, "
      "nameTableKind: Apple, sysroot: \"/\")\n"
      "!1 = !DIFile(filename: \"/tmp/code.cpp\", directory: "
      "\"/Users/shubham/Development/llvm-project/build_ninja\", checksumkind: "
      "CSK_MD5, checksum: \"719364c4b07176af8515cac6bd21008c\")\n"
      "!2 = !{i32 7, !\"Dwarf Version\", i32 5}\n"
      "!3 = !{i32 2, !\"Debug Info Version\", i32 3}\n"
      "!4 = !{i32 1, !\"wchar_size\", i32 4}\n"
      "!5 = !{i32 8, !\"PIC Level\", i32 2}\n"
      "!6 = !{i32 7, !\"uwtable\", i32 1}\n"
      "!7 = !{i32 7, !\"frame-pointer\", i32 1}\n"
      "!8 = !{!\"clang version 20.0.0git (git@github.com:llvm/llvm-project.git "
      "baff49d3a0ef8e0848a726656ebf6e7b310e5113)\"}\n"
      "!9 = distinct !DISubprogram(name: \"foo\", linkageName: \"_Z3fooi\", "
      "scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, flags: "
      "DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition "
      "| DISPFlagOptimized, unit: !0, retainedNodes: !14)\n"
      "!10 = !DIFile(filename: \"/tmp/code.cpp\", directory: \"\", "
      "checksumkind: CSK_MD5, checksum: \"719364c4b07176af8515cac6bd21008c\")\n"
      "!11 = !DISubroutineType(types: !12)\n"
      "!12 = !{!13, !13}\n"
      "!13 = !DIBasicType(name: \"int\", size: 32, encoding: DW_ATE_signed)\n"
      "!14 = !{!15}\n"
      "!15 = !DILocalVariable(name: \"x\", arg: 1, scope: !9, file: !10, line: "
      "1, type: !13)\n"
      "!16 = !DILocation(line: 0, scope: !9)\n"
      "!17 = !DILocation(line: 2, column: 11, scope: !18)\n"
      "!18 = distinct !DISubprogram(name: \"bar\", linkageName: \"_Z3bari\", "
      "scope: !10, file: !10, line: 11, type: !11, scopeLine: 1, flags: "
      "DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition "
      "| DISPFlagOptimized, unit: !0, retainedNodes: !14)\n";

  std::unique_ptr<llvm::Module> M = parseIR(C, IR);
  ASSERT_TRUE(M);

  DroppedVariableStats Stats(true);
  Stats.runBeforePass("Test",
                      llvm::Any(const_cast<const llvm::Module *>(M.get())));

  // Remove instructions
  for (auto &F : *M.get()) {
    for (auto &I : instructions(&F)) {
      I.dropDbgRecords();
      break;
    }
    break;
  }
  PreservedAnalyses PA;
  Stats.runAfterPass("Test",
                     llvm::Any(const_cast<const llvm::Module *>(M.get())), PA);
  ASSERT_EQ(Stats.getPassDroppedVariables(), false);
}

// This test ensures that if a #dbg_value is dropped after an optimization pass,
// but an instruction that has a scope which is a child of the #dbg_value scope
// still exists, debug information is conisdered dropped.
TEST(DroppedVariableStats, ChildScopes) {
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *IR =
      "; Function Attrs: mustprogress nounwind ssp uwtable(sync)\n"
      "define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 "
      "noundef %x) local_unnamed_addr #0 !dbg !9 {\n"
      "entry:\n"
      "#dbg_value(i32 %x, !15, !DIExpression(), !16)\n"
      "%add = add nsw i32 %x, 1, !dbg !17\n"
      "ret i32 0\n"
      "}\n"
      "!llvm.dbg.cu = !{!0}\n"
      "!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}\n"
      "!llvm.ident = !{!8}\n"
      "!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: "
      "!1, producer: \"clang version 20.0.0git "
      "(git@github.com:llvm/llvm-project.git "
      "baff49d3a0ef8e0848a726656ebf6e7b310e5113)\", isOptimized: true, "
      "runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, "
      "nameTableKind: Apple, sysroot: \"/\")\n"
      "!1 = !DIFile(filename: \"/tmp/code.cpp\", directory: "
      "\"/Users/shubham/Development/llvm-project/build_ninja\", checksumkind: "
      "CSK_MD5, checksum: \"719364c4b07176af8515cac6bd21008c\")\n"
      "!2 = !{i32 7, !\"Dwarf Version\", i32 5}\n"
      "!3 = !{i32 2, !\"Debug Info Version\", i32 3}\n"
      "!4 = !{i32 1, !\"wchar_size\", i32 4}\n"
      "!5 = !{i32 8, !\"PIC Level\", i32 2}\n"
      "!6 = !{i32 7, !\"uwtable\", i32 1}\n"
      "!7 = !{i32 7, !\"frame-pointer\", i32 1}\n"
      "!8 = !{!\"clang version 20.0.0git (git@github.com:llvm/llvm-project.git "
      "baff49d3a0ef8e0848a726656ebf6e7b310e5113)\"}\n"
      "!9 = distinct !DISubprogram(name: \"foo\", linkageName: \"_Z3fooi\", "
      "scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, flags: "
      "DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition "
      "| DISPFlagOptimized, unit: !0, retainedNodes: !14)\n"
      "!10 = !DIFile(filename: \"/tmp/code.cpp\", directory: \"\", "
      "checksumkind: CSK_MD5, checksum: \"719364c4b07176af8515cac6bd21008c\")\n"
      "!11 = !DISubroutineType(types: !12)\n"
      "!12 = !{!13, !13}\n"
      "!13 = !DIBasicType(name: \"int\", size: 32, encoding: DW_ATE_signed)\n"
      "!14 = !{!15}\n"
      "!15 = !DILocalVariable(name: \"x\", arg: 1, scope: !9, file: !10, line: "
      "1, type: !13)\n"
      "!16 = !DILocation(line: 0, scope: !9)\n"
      "!17 = !DILocation(line: 2, column: 11, scope: !18)\n"
      "!18 = distinct !DILexicalBlock(scope: !9, file: !10, line: 10, column: "
      "28)\n";

  std::unique_ptr<llvm::Module> M = parseIR(C, IR);
  ASSERT_TRUE(M);

  DroppedVariableStats Stats(true);
  Stats.runBeforePass("Test",
                      llvm::Any(const_cast<const llvm::Module *>(M.get())));

  // Remove instructions
  for (auto &F : *M.get()) {
    for (auto &I : instructions(&F)) {
      I.dropDbgRecords();
      break;
    }
    break;
  }
  PreservedAnalyses PA;
  Stats.runAfterPass("Test",
                     llvm::Any(const_cast<const llvm::Module *>(M.get())), PA);
  ASSERT_EQ(Stats.getPassDroppedVariables(), true);
}

// This test ensures that if a #dbg_value is dropped after an optimization pass,
// but an instruction that has a scope which is a child of the #dbg_value scope
// still exists, and the #dbg_value is inlined at another location, debug
// information is conisdered not dropped.
TEST(DroppedVariableStats, InlinedAt) {
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *IR =
      "; Function Attrs: mustprogress nounwind ssp uwtable(sync)\n"
      "define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 "
      "noundef %x) local_unnamed_addr #0 !dbg !9 {\n"
      "entry:\n"
      "#dbg_value(i32 %x, !15, !DIExpression(), !16)\n"
      "%add = add nsw i32 %x, 1, !dbg !17\n"
      "ret i32 0\n"
      "}\n"
      "!llvm.dbg.cu = !{!0}\n"
      "!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}\n"
      "!llvm.ident = !{!8}\n"
      "!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: "
      "!1, producer: \"clang version 20.0.0git "
      "(git@github.com:llvm/llvm-project.git "
      "baff49d3a0ef8e0848a726656ebf6e7b310e5113)\", isOptimized: true, "
      "runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, "
      "nameTableKind: Apple, sysroot: \"/\")\n"
      "!1 = !DIFile(filename: \"/tmp/code.cpp\", directory: "
      "\"/Users/shubham/Development/llvm-project/build_ninja\", checksumkind: "
      "CSK_MD5, checksum: \"719364c4b07176af8515cac6bd21008c\")\n"
      "!2 = !{i32 7, !\"Dwarf Version\", i32 5}\n"
      "!3 = !{i32 2, !\"Debug Info Version\", i32 3}\n"
      "!4 = !{i32 1, !\"wchar_size\", i32 4}\n"
      "!5 = !{i32 8, !\"PIC Level\", i32 2}\n"
      "!6 = !{i32 7, !\"uwtable\", i32 1}\n"
      "!7 = !{i32 7, !\"frame-pointer\", i32 1}\n"
      "!8 = !{!\"clang version 20.0.0git (git@github.com:llvm/llvm-project.git "
      "baff49d3a0ef8e0848a726656ebf6e7b310e5113)\"}\n"
      "!9 = distinct !DISubprogram(name: \"foo\", linkageName: \"_Z3fooi\", "
      "scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, flags: "
      "DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition "
      "| DISPFlagOptimized, unit: !0, retainedNodes: !14)\n"
      "!10 = !DIFile(filename: \"/tmp/code.cpp\", directory: \"\", "
      "checksumkind: CSK_MD5, checksum: \"719364c4b07176af8515cac6bd21008c\")\n"
      "!11 = !DISubroutineType(types: !12)\n"
      "!12 = !{!13, !13}\n"
      "!13 = !DIBasicType(name: \"int\", size: 32, encoding: DW_ATE_signed)\n"
      "!14 = !{!15}\n"
      "!15 = !DILocalVariable(name: \"x\", arg: 1, scope: !9, file: !10, line: "
      "1, type: !13)\n"
      "!16 = !DILocation(line: 0, scope: !9, inlinedAt: !19)\n"
      "!17 = !DILocation(line: 2, column: 11, scope: !18)\n"
      "!18 = distinct !DILexicalBlock(scope: !9, file: !10, line: 10, column: "
      "28)\n"
      "!19 = !DILocation(line: 3, column: 2, scope: !9)\n";

  std::unique_ptr<llvm::Module> M = parseIR(C, IR);
  ASSERT_TRUE(M);

  DroppedVariableStats Stats(true);
  Stats.runBeforePass("Test",
                      llvm::Any(const_cast<const llvm::Module *>(M.get())));

  // Remove instructions
  for (auto &F : *M.get()) {
    for (auto &I : instructions(&F)) {
      I.dropDbgRecords();
      break;
    }
    break;
  }
  PreservedAnalyses PA;
  Stats.runAfterPass("Test",
                     llvm::Any(const_cast<const llvm::Module *>(M.get())), PA);
  ASSERT_EQ(Stats.getPassDroppedVariables(), false);
}

// This test ensures that if a #dbg_value is dropped after an optimization pass,
// but an instruction that has a scope which is a child of the #dbg_value scope
// still exists, and the #dbg_value and the instruction are inlined at another
// location, debug information is conisdered dropped.
TEST(DroppedVariableStats, InlinedAtShared) {
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *IR =
      "; Function Attrs: mustprogress nounwind ssp uwtable(sync)\n"
      "define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 "
      "noundef %x) local_unnamed_addr #0 !dbg !9 {\n"
      "entry:\n"
      "#dbg_value(i32 %x, !15, !DIExpression(), !16)\n"
      "%add = add nsw i32 %x, 1, !dbg !17\n"
      "ret i32 0\n"
      "}\n"
      "!llvm.dbg.cu = !{!0}\n"
      "!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}\n"
      "!llvm.ident = !{!8}\n"
      "!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: "
      "!1, producer: \"clang version 20.0.0git "
      "(git@github.com:llvm/llvm-project.git "
      "baff49d3a0ef8e0848a726656ebf6e7b310e5113)\", isOptimized: true, "
      "runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, "
      "nameTableKind: Apple, sysroot: \"/\")\n"
      "!1 = !DIFile(filename: \"/tmp/code.cpp\", directory: "
      "\"/Users/shubham/Development/llvm-project/build_ninja\", checksumkind: "
      "CSK_MD5, checksum: \"719364c4b07176af8515cac6bd21008c\")\n"
      "!2 = !{i32 7, !\"Dwarf Version\", i32 5}\n"
      "!3 = !{i32 2, !\"Debug Info Version\", i32 3}\n"
      "!4 = !{i32 1, !\"wchar_size\", i32 4}\n"
      "!5 = !{i32 8, !\"PIC Level\", i32 2}\n"
      "!6 = !{i32 7, !\"uwtable\", i32 1}\n"
      "!7 = !{i32 7, !\"frame-pointer\", i32 1}\n"
      "!8 = !{!\"clang version 20.0.0git (git@github.com:llvm/llvm-project.git "
      "baff49d3a0ef8e0848a726656ebf6e7b310e5113)\"}\n"
      "!9 = distinct !DISubprogram(name: \"foo\", linkageName: \"_Z3fooi\", "
      "scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, flags: "
      "DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition "
      "| DISPFlagOptimized, unit: !0, retainedNodes: !14)\n"
      "!10 = !DIFile(filename: \"/tmp/code.cpp\", directory: \"\", "
      "checksumkind: CSK_MD5, checksum: \"719364c4b07176af8515cac6bd21008c\")\n"
      "!11 = !DISubroutineType(types: !12)\n"
      "!12 = !{!13, !13}\n"
      "!13 = !DIBasicType(name: \"int\", size: 32, encoding: DW_ATE_signed)\n"
      "!14 = !{!15}\n"
      "!15 = !DILocalVariable(name: \"x\", arg: 1, scope: !9, file: !10, line: "
      "1, type: !13)\n"
      "!16 = !DILocation(line: 0, scope: !9, inlinedAt: !19)\n"
      "!17 = !DILocation(line: 2, column: 11, scope: !18, inlinedAt: !19)\n"
      "!18 = distinct !DILexicalBlock(scope: !9, file: !10, line: 10, column: "
      "28)\n"
      "!19 = !DILocation(line: 3, column: 2, scope: !9)\n";

  std::unique_ptr<llvm::Module> M = parseIR(C, IR);
  ASSERT_TRUE(M);

  DroppedVariableStats Stats(true);
  Stats.runBeforePass("Test",
                      llvm::Any(const_cast<const llvm::Module *>(M.get())));

  // Remove instructions
  for (auto &F : *M.get()) {
    for (auto &I : instructions(&F)) {
      I.dropDbgRecords();
      break;
    }
    break;
  }
  PreservedAnalyses PA;
  Stats.runAfterPass("Test",
                     llvm::Any(const_cast<const llvm::Module *>(M.get())), PA);
  ASSERT_EQ(Stats.getPassDroppedVariables(), true);
}

// This test ensures that if a #dbg_value is dropped after an optimization pass,
// but an instruction that has a scope which is a child of the #dbg_value scope
// still exists, and the instruction is inlined at a location that is the
// #dbg_value's inlined at location, debug information is conisdered dropped.
TEST(DroppedVariableStats, InlinedAtChild) {
  PassInstrumentationCallbacks PIC;
  PassInstrumentation PI(&PIC);

  LLVMContext C;

  const char *IR =
      "; Function Attrs: mustprogress nounwind ssp uwtable(sync)\n"
      "define noundef range(i32 -2147483647, -2147483648) i32 @_Z3fooi(i32 "
      "noundef %x) local_unnamed_addr #0 !dbg !9 {\n"
      "entry:\n"
      "#dbg_value(i32 %x, !15, !DIExpression(), !16)\n"
      "%add = add nsw i32 %x, 1, !dbg !17\n"
      "ret i32 0\n"
      "}\n"
      "!llvm.dbg.cu = !{!0}\n"
      "!llvm.module.flags = !{!2, !3, !4, !5, !6, !7}\n"
      "!llvm.ident = !{!8}\n"
      "!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: "
      "!1, producer: \"clang version 20.0.0git "
      "(git@github.com:llvm/llvm-project.git "
      "baff49d3a0ef8e0848a726656ebf6e7b310e5113)\", isOptimized: true, "
      "runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, "
      "nameTableKind: Apple, sysroot: \"/\")\n"
      "!1 = !DIFile(filename: \"/tmp/code.cpp\", directory: "
      "\"/Users/shubham/Development/llvm-project/build_ninja\", checksumkind: "
      "CSK_MD5, checksum: \"719364c4b07176af8515cac6bd21008c\")\n"
      "!2 = !{i32 7, !\"Dwarf Version\", i32 5}\n"
      "!3 = !{i32 2, !\"Debug Info Version\", i32 3}\n"
      "!4 = !{i32 1, !\"wchar_size\", i32 4}\n"
      "!5 = !{i32 8, !\"PIC Level\", i32 2}\n"
      "!6 = !{i32 7, !\"uwtable\", i32 1}\n"
      "!7 = !{i32 7, !\"frame-pointer\", i32 1}\n"
      "!8 = !{!\"clang version 20.0.0git (git@github.com:llvm/llvm-project.git "
      "baff49d3a0ef8e0848a726656ebf6e7b310e5113)\"}\n"
      "!9 = distinct !DISubprogram(name: \"foo\", linkageName: \"_Z3fooi\", "
      "scope: !10, file: !10, line: 1, type: !11, scopeLine: 1, flags: "
      "DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition "
      "| DISPFlagOptimized, unit: !0, retainedNodes: !14)\n"
      "!10 = !DIFile(filename: \"/tmp/code.cpp\", directory: \"\", "
      "checksumkind: CSK_MD5, checksum: \"719364c4b07176af8515cac6bd21008c\")\n"
      "!11 = !DISubroutineType(types: !12)\n"
      "!12 = !{!13, !13}\n"
      "!13 = !DIBasicType(name: \"int\", size: 32, encoding: DW_ATE_signed)\n"
      "!14 = !{!15}\n"
      "!15 = !DILocalVariable(name: \"x\", arg: 1, scope: !9, file: !10, line: "
      "1, type: !13)\n"
      "!16 = !DILocation(line: 0, scope: !9, inlinedAt: !19)\n"
      "!17 = !DILocation(line: 2, column: 11, scope: !18, inlinedAt: !20)\n"
      "!18 = distinct !DILexicalBlock(scope: !9, file: !10, line: 10, column: "
      "28)\n"
      "!19 = !DILocation(line: 3, column: 2, scope: !9)\n"
      "!20 = !DILocation(line: 4, column: 5, scope: !18, inlinedAt: !19)";

  std::unique_ptr<llvm::Module> M = parseIR(C, IR);
  ASSERT_TRUE(M);

  DroppedVariableStats Stats(true);
  Stats.runBeforePass("Test",
                      llvm::Any(const_cast<const llvm::Module *>(M.get())));

  // Remove instructions
  for (auto &F : *M.get()) {
    for (auto &I : instructions(&F)) {
      I.dropDbgRecords();
      break;
    }
    break;
  }
  PreservedAnalyses PA;
  Stats.runAfterPass("Test",
                     llvm::Any(const_cast<const llvm::Module *>(M.get())), PA);
  ASSERT_EQ(Stats.getPassDroppedVariables(), true);
}

} // end anonymous namespace
