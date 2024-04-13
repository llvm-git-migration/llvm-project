//===-- StringCompareCheck.cpp - clang-tidy--------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StringCompareCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Tooling/FixIt.h"
#include "llvm/ADT/StringRef.h"

using namespace clang::ast_matchers;
namespace optutils = clang::tidy::utils::options;

namespace clang::tidy::readability {

static const StringRef CompareMessage = "do not use 'compare' to test equality "
                                        "of strings; use the string equality "
                                        "operator instead";

static const StringRef DefaultStringClassNames = "::std::basic_string;"
                                                 "::std::basic_string_view";

StringCompareCheck::StringCompareCheck(StringRef Name,
                                       ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      StringClassNames(optutils::parseStringList(
          Options.get("StringClassNames", DefaultStringClassNames))),
      // Both std::string and std::string_view are templates, so this check only
      // needs to match template classes by default.
      // Custom `StringClassNames` may contain non-template classes, and
      // it's impossible to tell them apart from templates just by name.
      CheckNonTemplateClasses(Options.get("StringClassNames").has_value()) {}

void StringCompareCheck::registerMatchers(MatchFinder *Finder) {
  if (StringClassNames.empty()) {
    return;
  }
  const auto RegisterForClasses = [&, this](const auto &StringClassMatcher) {
    const auto StrCompare = cxxMemberCallExpr(
        callee(cxxMethodDecl(hasName("compare"), ofClass(StringClassMatcher))),
        hasArgument(0, expr().bind("str2")), argumentCountIs(1),
        callee(memberExpr().bind("str1")));

    // First and second case: cast str.compare(str) to boolean.
    Finder->addMatcher(
        traverse(TK_AsIs,
                 implicitCastExpr(hasImplicitDestinationType(booleanType()),
                                  has(StrCompare))
                     .bind("match1")),
        this);

    // Third and fourth case: str.compare(str) == 0
    // and str.compare(str) !=  0.
    Finder->addMatcher(
        binaryOperator(hasAnyOperatorName("==", "!="),
                       hasOperands(StrCompare.bind("compare"),
                                   integerLiteral(equals(0)).bind("zero")))
            .bind("match2"),
        this);
  };
  auto TemplateClassMatcher =
      classTemplateSpecializationDecl(hasAnyName(StringClassNames));
  if (CheckNonTemplateClasses) {
    RegisterForClasses(anyOf(std::move(TemplateClassMatcher),
                             cxxRecordDecl(hasAnyName(StringClassNames))));
  } else {
    RegisterForClasses(std::move(TemplateClassMatcher));
  }
}

void StringCompareCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *Matched = Result.Nodes.getNodeAs<Stmt>("match1")) {
    diag(Matched->getBeginLoc(), CompareMessage);
    return;
  }

  if (const auto *Matched = Result.Nodes.getNodeAs<Stmt>("match2")) {
    const ASTContext &Ctx = *Result.Context;

    if (const auto *Zero = Result.Nodes.getNodeAs<Stmt>("zero")) {
      const auto *Str1 = Result.Nodes.getNodeAs<MemberExpr>("str1");
      const auto *Str2 = Result.Nodes.getNodeAs<Stmt>("str2");
      const auto *Compare = Result.Nodes.getNodeAs<Stmt>("compare");

      auto Diag = diag(Matched->getBeginLoc(), CompareMessage);

      if (Str1->isArrow())
        Diag << FixItHint::CreateInsertion(Str1->getBeginLoc(), "*");

      Diag << tooling::fixit::createReplacement(*Zero, *Str2, Ctx)
           << tooling::fixit::createReplacement(*Compare, *Str1->getBase(),
                                                Ctx);
    }
  }

  // FIXME: Add fixit to fix the code for case one and two (match1).
}

} // namespace clang::tidy::readability
