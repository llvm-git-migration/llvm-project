//===--- NondetermnisticPointerUsageCheck.cpp - clang-tidy ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "NondeterministicPointerUsageCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

void NondeterministicPointerUsageCheck::registerMatchers(MatchFinder *Finder) {

  auto LoopVariable = varDecl(hasType(hasCanonicalType(pointerType())));

  auto RangeInit = declRefExpr(to(varDecl(hasType(recordDecl(
      anyOf(hasName("std::unordered_set"), hasName("std::unordered_map"),
            hasName("std::unordered_multiset"),
            hasName("std::unordered_multimap")))))));

  Finder->addMatcher(
      stmt(cxxForRangeStmt(hasRangeInit(RangeInit.bind("rangeinit")),
                           hasLoopVariable(LoopVariable.bind("loopVar"))))
          .bind("cxxForRangeStmt"),
      this);

  auto SortFuncM = anyOf(callee(functionDecl(hasName("std::is_sorted"))),
                         callee(functionDecl(hasName("std::nth_element"))),
                         callee(functionDecl(hasName("std::sort"))),
                         callee(functionDecl(hasName("std::partial_sort"))),
                         callee(functionDecl(hasName("std::partition"))),
                         callee(functionDecl(hasName("std::stable_partition"))),
                         callee(functionDecl(hasName("std::stable_sort"))));

  auto IteratesPointerEltsM = hasArgument(
      0,
      cxxMemberCallExpr(on(hasType(cxxRecordDecl(has(fieldDecl(hasType(
          hasCanonicalType(pointsTo(hasCanonicalType(pointerType())))))))))));

  Finder->addMatcher(stmt(callExpr(allOf(SortFuncM, IteratesPointerEltsM)))
                         .bind("sortsemantic"),
                     this);
}

void NondeterministicPointerUsageCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *ForRangePointers =
      Result.Nodes.getNodeAs<Stmt>("cxxForRangeStmt");
  const auto *SortPointers = Result.Nodes.getNodeAs<Stmt>("sortsemantic");

  if ((ForRangePointers) && !(ForRangePointers->getBeginLoc().isMacroID())) {
    const auto *Node = dyn_cast<CXXForRangeStmt>(ForRangePointers);
    diag(Node->getRParenLoc(), "Iteration of pointers is nondeterministic");
  }

  if ((SortPointers) && !(SortPointers->getBeginLoc().isMacroID())) {
    const auto *Node = dyn_cast<Stmt>(SortPointers);
    diag(Node->getBeginLoc(), "Sorting pointers is nondeterministic");
  }
}

} // namespace clang::tidy::bugprone
