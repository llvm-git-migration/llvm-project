//===--- UndefinedSprintfOverlapCheck.cpp - clang-tidy --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UndefinedSprintfOverlapCheck.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

AST_MATCHER_P(CallExpr, hasAnyOtherArgument,
              ast_matchers::internal::Matcher<Expr>, InnerMatcher) {
  for (const auto *Arg : llvm::drop_begin(Node.arguments())) {
    ast_matchers::internal::BoundNodesTreeBuilder Result(*Builder);
    if (InnerMatcher.matches(*Arg, Finder, &Result)) {
      *Builder = std::move(Result);
      return true;
    }
  }
  return false;
}

void UndefinedSprintfOverlapCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(
          callee(
              functionDecl(matchesName("(::std)?::(sn?printf)")).bind("decl")),
          hasArgument(0, ignoringParenImpCasts(
                             declRefExpr(to(varDecl().bind("firstArg"))))),
          hasAnyOtherArgument(
              ignoringParenImpCasts(declRefExpr().bind("overlappingArg")))),
      this);
}

void UndefinedSprintfOverlapCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *OverlappingArg =
      Result.Nodes.getNodeAs<DeclRefExpr>("overlappingArg");
  const auto *FirstArg = Result.Nodes.getNodeAs<VarDecl>("firstArg");
  const auto *FnDecl = Result.Nodes.getNodeAs<FunctionDecl>("decl");

  diag(OverlappingArg->getLocation(), "argument %0 overlaps the first argument "
                                      "in %1, which is undefined behavior")
      << FirstArg << FnDecl;
}

} // namespace clang::tidy::bugprone
