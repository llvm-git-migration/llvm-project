//===--- SubstrToStartsWithCheck.cpp - clang-tidy ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SubstrToStartsWithCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

void SubstrToStartsWithCheck::registerMatchers(MatchFinder *Finder) {
  auto isZeroExpr = expr(anyOf(
      integerLiteral(equals(0)),
      ignoringParenImpCasts(declRefExpr(
          to(varDecl(hasInitializer(integerLiteral(equals(0))))))),
      binaryOperator(hasOperatorName("-"), hasLHS(expr()), hasRHS(expr()))));

  auto isStringLike = expr(anyOf(
      stringLiteral().bind("literal"),
      implicitCastExpr(hasSourceExpression(stringLiteral().bind("literal"))),
      declRefExpr(to(varDecl(hasType(qualType(hasDeclaration(
          namedDecl(hasAnyName("::std::string", "::std::basic_string")))))))).bind("strvar")));

  auto isSubstrCall = 
      cxxMemberCallExpr(
          callee(memberExpr(hasDeclaration(cxxMethodDecl(
              hasName("substr"),
              ofClass(hasAnyName("basic_string", "string", "u16string")))))),
          hasArgument(0, isZeroExpr),
          hasArgument(1, expr().bind("length")))
          .bind("substr");

  Finder->addMatcher(
      binaryOperator(
          anyOf(hasOperatorName("=="), hasOperatorName("!=")),
          hasEitherOperand(isSubstrCall),
          hasEitherOperand(isStringLike),
          unless(hasType(isAnyCharacter())))
          .bind("comparison"),
      this);

  Finder->addMatcher(
      cxxMemberCallExpr(
          callee(memberExpr(hasDeclaration(cxxMethodDecl(
              hasName("substr"),
              ofClass(hasAnyName("basic_string", "string", "u16string")))))),
          hasArgument(0, isZeroExpr),
          hasArgument(1, expr().bind("direct_length")))
          .bind("direct_substr"),
      this);
}

void SubstrToStartsWithCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Comparison = Result.Nodes.getNodeAs<BinaryOperator>("comparison");

  if (Comparison) {
    const auto *SubstrCall = Result.Nodes.getNodeAs<CXXMemberCallExpr>("substr");
    const auto *LengthArg = Result.Nodes.getNodeAs<Expr>("length");
    const auto *Literal = Result.Nodes.getNodeAs<StringLiteral>("literal");
    const auto *StrVar = Result.Nodes.getNodeAs<DeclRefExpr>("strvar");

    if (!SubstrCall || !LengthArg || (!Literal && !StrVar))
      return;

    std::string CompareStr;
    if (Literal) {
      CompareStr = Literal->getString().str();
    } else if (StrVar) {
      CompareStr = Lexer::getSourceText(
          CharSourceRange::getTokenRange(StrVar->getSourceRange()),
          *Result.SourceManager, Result.Context->getLangOpts())
          .str();
    }

    if (Literal) {
      if (const auto *LengthLiteral = dyn_cast<IntegerLiteral>(LengthArg)) {
        if (LengthLiteral->getValue() != Literal->getLength())
          return;
      }
    }

    std::string Replacement;
    if (Comparison->getOpcode() == BO_EQ) {
      Replacement = "starts_with(" + CompareStr + ")";
    } else { // BO_NE
      Replacement = "!starts_with(" + CompareStr + ")";
    }

    diag(Comparison->getBeginLoc(),
         "use starts_with() instead of substring comparison")
        << FixItHint::CreateReplacement(Comparison->getSourceRange(),
                                      Replacement);
  }
}

} // namespace clang::tidy::modernize
