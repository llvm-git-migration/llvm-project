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
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/StringMap.h"

namespace clang::tidy::utils {

/// FIXME: Base class for handling converting std iterator algorithms to a range
/// equivalent
class UseRangesCheck : public ClangTidyCheck {
public:
  class Replacer : public llvm::RefCountedBase<Replacer> {
  public:
    struct Indexes {
      enum Replace { First, Second };
      unsigned BeginArg;
      unsigned EndArg = BeginArg + 1;
      Replace ReplaceArg = First;
    };

    virtual std::string getReplaceName(const NamedDecl &OriginalName) const = 0;
    virtual std::optional<std::string>
    getHeaderInclusion(const NamedDecl &OriginalName) const;
    virtual ArrayRef<ArrayRef<Indexes>> getReplacementSignatures() const = 0;
    virtual ~Replacer() = default;
  };

  using ReplacerMap = llvm::StringMap<llvm::IntrusiveRefCntPtr<Replacer>>;

  UseRangesCheck(StringRef Name, ClangTidyContext *Context);
  virtual ReplacerMap GetReplacerMap() const = 0;
  void registerPPCallbacks(const SourceManager &SM, Preprocessor *PP,
                           Preprocessor *ModuleExpanderPP) override;
  void registerMatchers(ast_matchers::MatchFinder *Finder) final;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) final;
  bool isLanguageVersionSupported(const LangOptions &LangOpts) const override;
  void storeOptions(ClangTidyOptions::OptionMap &Options) override;

private:
  ReplacerMap Replaces;
  IncludeInserter Inserter;
};

} // namespace clang::tidy::utils

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_UTILS_USERANGESCHECK_H
