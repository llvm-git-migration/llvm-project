//===--- MarkStaticCheck.cpp - clang-tidy ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MarkStaticCheck.h"
#include "clang/AST/Decl.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Basic/Specifiers.h"

using namespace clang::ast_matchers;

namespace clang::tidy::readability {

namespace {

AST_POLYMORPHIC_MATCHER(isFirstDecl,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl,
                                                        VarDecl)) {
  return Node.isFirstDecl();
}

AST_MATCHER(Decl, isInMainFile) {
  return Finder->getASTContext().getSourceManager().isInMainFile(
      Node.getLocation());
}

AST_POLYMORPHIC_MATCHER(isExternStorageClass,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(FunctionDecl,
                                                        VarDecl)) {
  return Node.getStorageClass() == SC_Extern;
}

} // namespace

void MarkStaticCheck::registerMatchers(MatchFinder *Finder) {
  auto Common =
      allOf(isFirstDecl(), isInMainFile(),
            unless(anyOf(isStaticStorageClass(), isExternStorageClass(),
                         isInAnonymousNamespace())));
  Finder->addMatcher(
      functionDecl(Common, unless(cxxMethodDecl()), unless(isMain()))
          .bind("fn"),
      this);
  Finder->addMatcher(varDecl(Common, hasGlobalStorage()).bind("var"), this);
}

void MarkStaticCheck::check(const MatchFinder::MatchResult &Result) {
  if (const auto *FD = Result.Nodes.getNodeAs<FunctionDecl>("fn")) {
    diag(FD->getLocation(), "function %0 can be static") << FD;
    return;
  }
  if (const auto *VD = Result.Nodes.getNodeAs<VarDecl>("var")) {
    diag(VD->getLocation(), "variable %0 can be static") << VD;
    return;
  }
  llvm_unreachable("");
}

} // namespace clang::tidy::readability
