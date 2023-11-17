//===--- RedundantInlineSpecifierCheck.cpp -
// clang-tidy----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RedundantInlineSpecifierCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"

#include "../utils/LexerUtils.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

static std::optional<SourceLocation>
getInlineTokenLocation(SourceRange RangeLocation, const SourceManager &Sources,
                       const LangOptions &LangOpts) {
  SourceLocation CurrentLocation = RangeLocation.getBegin();
  Token CurrentToken;
  while (!Lexer::getRawToken(CurrentLocation, CurrentToken, Sources, LangOpts,
                             true) &&
         CurrentLocation < RangeLocation.getEnd() &&
         CurrentToken.isNot(tok::eof)) {
    if (CurrentToken.is(tok::raw_identifier)) {
      if (CurrentToken.getRawIdentifier() == "inline") {
        return CurrentToken.getLocation();
      }
    }
    CurrentLocation = CurrentToken.getEndLoc();
  }
  return std::nullopt;
}

void RedundantInlineSpecifierCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      functionDecl(unless(isExpansionInSystemHeader()), unless(isImplicit()),
                   unless(hasAncestor(lambdaExpr())), isInline(),
                   anyOf(isConstexpr(), isDeleted(),
                         allOf(isDefinition(), hasAncestor(recordDecl()),
                               unless(hasAncestor(cxxConstructorDecl())))))
          .bind("fun_decl"),
      this);

  Finder->addMatcher(
      varDecl(isInline(), unless(isImplicit()),
              anyOf(allOf(isConstexpr(), unless(isStaticStorageClass())),
                    hasAncestor(recordDecl())))
          .bind("var_decl"),
      this);

  Finder->addMatcher(
      functionTemplateDecl(has(functionDecl(isInline()))).bind("templ_decl"),
      this);
}

template <typename T>
void RedundantInlineSpecifierCheck::handleMatchedDecl(
    const T *MatchedDecl, const SourceManager &Sources,
    const MatchFinder::MatchResult &Result, const char *Message) {
  if (auto Loc = getInlineTokenLocation(MatchedDecl->getSourceRange(), Sources,
                                        Result.Context->getLangOpts()))
    diag(*Loc, Message) << MatchedDecl << FixItHint::CreateRemoval(*Loc);
}

void RedundantInlineSpecifierCheck::check(
    const MatchFinder::MatchResult &Result) {
  const SourceManager &Sources = *Result.SourceManager;

  if (const auto *MatchedDecl =
          Result.Nodes.getNodeAs<FunctionDecl>("fun_decl")) {
    handleMatchedDecl(
        MatchedDecl, Sources, Result,
        "Function %0 has inline specifier but is implicitly inlined");
  } else if (const auto *MatchedDecl =
                 Result.Nodes.getNodeAs<VarDecl>("var_decl")) {
    handleMatchedDecl(
        MatchedDecl, Sources, Result,
        "Variable %0 has inline specifier but is implicitly inlined");
  } else if (const auto *MatchedDecl =
                 Result.Nodes.getNodeAs<FunctionTemplateDecl>("templ_decl")) {
    handleMatchedDecl(
        MatchedDecl, Sources, Result,
        "Function %0 has inline specifier but is implicitly inlined");
  }
}

} // namespace clang::tidy::readability
