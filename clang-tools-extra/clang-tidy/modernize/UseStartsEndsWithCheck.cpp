//===--- UseStartsEndsWithCheck.cpp - clang-tidy --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseStartsEndsWithCheck.h"

#include "../utils/OptionsUtils.h"
#include "clang/Lex/Lexer.h"

#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {
namespace {
// Given two argument indices X and Y, matches when a call expression has a
// string at index X with an expression representing that string's length at
// index Y. The string can be a string literal or a variable. The length can be
// matched via an integer literal or a call to strlen() in the case of a string
// literal, and by a call to size() or length() in the string variable case.
AST_POLYMORPHIC_MATCHER_P2(HasStringAndLengthArguments,
                           AST_POLYMORPHIC_SUPPORTED_TYPES(
                               CallExpr, CXXConstructExpr,
                               CXXUnresolvedConstructExpr, ObjCMessageExpr),
                           unsigned, StringArgIndex, unsigned, LengthArgIndex) {
  if (StringArgIndex >= Node.getNumArgs() ||
      LengthArgIndex >= Node.getNumArgs()) {
    return false;
  }

  const Expr *StringArgExpr =
      Node.getArg(StringArgIndex)->IgnoreParenImpCasts();
  const Expr *LengthArgExpr =
      Node.getArg(LengthArgIndex)->IgnoreParenImpCasts();

  if (const auto *StringArg = dyn_cast<StringLiteral>(StringArgExpr)) {
    // Match an integer literal equal to the string length or a call to strlen.
    const auto Matcher = expr(anyOf(
        integerLiteral(equals(StringArg->getLength())),
        callExpr(
            callee(functionDecl(hasName("strlen"))), argumentCountIs(1),
            hasArgument(0, stringLiteral(hasSize(StringArg->getLength()))))));
    return Matcher.matches(*LengthArgExpr, Finder, Builder);
  }

  if (const auto *StringArg = dyn_cast<DeclRefExpr>(StringArgExpr)) {
    // Match a call to size() or length() on the same variable.
    const auto Matcher = cxxMemberCallExpr(
        on(declRefExpr(to(varDecl(equalsNode(StringArg->getDecl()))))),
        callee(cxxMethodDecl(hasAnyName("size", "length"), isConst(),
                             parameterCountIs(0))));
    return Matcher.matches(*LengthArgExpr, Finder, Builder);
  }

  return false;
}
} // namespace

UseStartsEndsWithCheck::UseStartsEndsWithCheck(StringRef Name,
                                               ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void UseStartsEndsWithCheck::registerMatchers(MatchFinder *Finder) {
  const auto ZeroLiteral = integerLiteral(equals(0));
  const auto HasStartsWithMethodWithName = [](const std::string &Name) {
    return hasMethod(
        cxxMethodDecl(hasName(Name), isConst(), parameterCountIs(1))
            .bind("starts_with_fun"));
  };
  const auto HasStartsWithMethod =
      anyOf(HasStartsWithMethodWithName("starts_with"),
            HasStartsWithMethodWithName("startsWith"),
            HasStartsWithMethodWithName("startswith"));
  const auto ClassWithStartsWithFunction = cxxRecordDecl(anyOf(
      HasStartsWithMethod, hasAnyBase(hasType(hasCanonicalType(hasDeclaration(
                               cxxRecordDecl(HasStartsWithMethod)))))));

  const auto FindExpr = cxxMemberCallExpr(
      // A method call with no second argument or the second argument is zero...
      anyOf(argumentCountIs(1), hasArgument(1, ZeroLiteral)),
      // ... named find...
      callee(cxxMethodDecl(hasName("find")).bind("find_fun")),
      // ... on a class with a starts_with function.
      on(hasType(
          hasCanonicalType(hasDeclaration(ClassWithStartsWithFunction)))),
      // Bind search expression.
      hasArgument(0, expr().bind("search_expr")));

  const auto RFindExpr = cxxMemberCallExpr(
      // A method call with a second argument of zero...
      hasArgument(1, ZeroLiteral),
      // ... named rfind...
      callee(cxxMethodDecl(hasName("rfind")).bind("find_fun")),
      // ... on a class with a starts_with function.
      on(hasType(
          hasCanonicalType(hasDeclaration(ClassWithStartsWithFunction)))),
      // Bind search expression.
      hasArgument(0, expr().bind("search_expr")));

  const auto CompareExpr = cxxMemberCallExpr(
      // A method call with a first argument of zero...
      hasArgument(0, ZeroLiteral),
      // ... named compare...
      callee(cxxMethodDecl(hasName("compare")).bind("find_fun")),
      // ... on a class with a starts_with function...
      on(hasType(
          hasCanonicalType(hasDeclaration(ClassWithStartsWithFunction)))),
      // ... where the third argument is some string and the second its length.
      HasStringAndLengthArguments(2, 1),
      // Bind search expression.
      hasArgument(2, expr().bind("search_expr")));

  Finder->addMatcher(
      // Match [=!]= with a zero on one side and (r?)find|compare on the other.
      binaryOperator(
          hasAnyOperatorName("==", "!="),
          hasOperands(cxxMemberCallExpr(anyOf(FindExpr, RFindExpr, CompareExpr))
                          .bind("find_expr"),
                      ZeroLiteral))
          .bind("expr"),
      this);
}

void UseStartsEndsWithCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *ComparisonExpr = Result.Nodes.getNodeAs<BinaryOperator>("expr");
  const auto *FindExpr = Result.Nodes.getNodeAs<CXXMemberCallExpr>("find_expr");
  const auto *FindFun = Result.Nodes.getNodeAs<CXXMethodDecl>("find_fun");
  const auto *SearchExpr = Result.Nodes.getNodeAs<Expr>("search_expr");
  const auto *StartsWithFunction =
      Result.Nodes.getNodeAs<CXXMethodDecl>("starts_with_fun");

  if (ComparisonExpr->getBeginLoc().isMacroID()) {
    return;
  }

  const bool Neg = ComparisonExpr->getOpcode() == BO_NE;

  auto Diagnostic =
      diag(FindExpr->getExprLoc(), "use %0 instead of %1() %select{==|!=}2 0")
      << StartsWithFunction->getName() << FindFun->getName() << Neg;

  // Remove possible arguments after search expression and ' [!=]= 0' suffix.
  Diagnostic << FixItHint::CreateReplacement(
      CharSourceRange::getTokenRange(
          Lexer::getLocForEndOfToken(SearchExpr->getEndLoc(), 0,
                                     *Result.SourceManager, getLangOpts()),
          ComparisonExpr->getEndLoc()),
      ")");

  // Remove possible '0 [!=]= ' prefix.
  Diagnostic << FixItHint::CreateRemoval(CharSourceRange::getCharRange(
      ComparisonExpr->getBeginLoc(), FindExpr->getBeginLoc()));

  // Replace method name by 'starts_with'.
  // Remove possible arguments before search expression.
  Diagnostic << FixItHint::CreateReplacement(
      CharSourceRange::getCharRange(FindExpr->getExprLoc(),
                                    SearchExpr->getBeginLoc()),
      StartsWithFunction->getNameAsString() + "(");

  // Add possible negation '!'.
  if (Neg) {
    Diagnostic << FixItHint::CreateInsertion(FindExpr->getBeginLoc(), "!");
  }
}

} // namespace clang::tidy::modernize
