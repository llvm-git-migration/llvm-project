//===--- SubstrToStartsWithCheck.h - clang-tidy ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_SUBSTRTOSTARTSWITHCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_SUBSTRTOSTARTSWITHCHECK_H

#include "../ClangTidyCheck.h"

namespace clang::tidy::modernize {

/// Finds calls to substr(0, n) that can be replaced with starts_with().
///
/// For the user-facing documentation see:
/// http://clang.llvm.org/extra/clang-tidy/checks/modernize/substr-to-starts-with.html
class SubstrToStartsWithCheck : public ClangTidyCheck {
public:
  SubstrToStartsWithCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}
  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace clang::tidy::modernize

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MODERNIZE_SUBSTRTOSTARTSWITHCHECK_H
