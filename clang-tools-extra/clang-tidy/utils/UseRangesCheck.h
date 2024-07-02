//===--- UseRangesCheck.h - clang-tidy --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_USERANGESCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_USERANGESCHECK_H

#include "../ClangTidyCheck.h"
#include "IncludeInserter.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringMap.h"

namespace clang::tidy::utils {

/// Base class for handling converting std iterator algorithms to a range
/// equivalent.
class UseRangesCheck : public ClangTidyCheck {
public:
  struct Indexes {
    enum Replace { First, Second };
    unsigned BeginArg;
    unsigned EndArg = BeginArg + 1;
    Replace ReplaceArg = First;
  };

  using Signature = SmallVector<Indexes, 2>;

  class Replacer : public llvm::RefCountedBase<Replacer> {
  public:
    virtual std::string getReplaceName(const NamedDecl &OriginalName) const = 0;
    virtual std::optional<std::string>
    getHeaderInclusion(const NamedDecl &OriginalName) const;
    virtual ArrayRef<Signature> getReplacementSignatures() const = 0;
    virtual ~Replacer() = default;
  };

  using ReplacerMap = llvm::StringMap<llvm::IntrusiveRefCntPtr<Replacer>>;

  UseRangesCheck(StringRef Name, ClangTidyContext *Context);
  /// Gets a map of function to replace and methods to create the replacements
  virtual ReplacerMap getReplacerMap() const = 0;
  /// Create a diagnostic for the CallExpr
  /// Override this to support custom diagnostic messages
  virtual DiagnosticBuilder createDiag(const CallExpr &Call);

  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) final;
  void registerMatchers(ast_matchers::MatchFinder *Finder) final;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) final;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;
  void storeOptions(ClangTidyOptions::OptionMap &Options) override;
  std::optional<TraversalKind> getCheckTraversalKind() const override;

private:
  ReplacerMap Replaces;
  IncludeInserter Inserter;
};

} // namespace clang::tidy::utils

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_USERANGESCHECK_H
