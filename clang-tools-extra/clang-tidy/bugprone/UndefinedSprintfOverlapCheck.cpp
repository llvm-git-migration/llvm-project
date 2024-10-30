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

UndefinedSprintfOverlapCheck::UndefinedSprintfOverlapCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      SprintfRegex(Options.get("SprintfFunction", "(::std)?::(sn?printf)")) {}

void UndefinedSprintfOverlapCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(callee(functionDecl(matchesName(SprintfRegex)).bind("decl")),
               hasArgument(0, ignoringParenImpCasts(
                                  declRefExpr(to(varDecl().bind("firstArg"))))),
               hasAnyOtherArgument(ignoringParenImpCasts(
                   declRefExpr(to(varDecl(equalsBoundNode("firstArg"))))
                       .bind("overlappingArg")))),
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

void UndefinedSprintfOverlapCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "SprintfRegex", SprintfRegex);
}

} // namespace clang::tidy::bugprone
