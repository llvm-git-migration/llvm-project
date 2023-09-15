//===--- CodeSnippetHighlighter.h - Code snippet highlighting ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_FRONTEND_CODESNIPPETHIGHLIGHTER_H
#define LLVM_CLANG_FRONTEND_CODESNIPPETHIGHLIGHTER_H

#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>

namespace clang {

struct StyleRange {
  unsigned Start;
  unsigned End;
  const enum llvm::raw_ostream::Colors c;
};

class CodeSnippetHighlighter final {
public:
  CodeSnippetHighlighter() = default;

  /// Produce StyleRanges for the given line.
  /// The returned vector contains non-overlapping style ranges. They are sorted
  /// from beginning of the line to the end.
  std::vector<StyleRange> highlightLine(llvm::StringRef SourceLine,
                                        const LangOptions &LangOpts);

private:
  bool Initialized = false;
  /// Fills Keywords and Literals.
  void ensureTokenData();

  llvm::SmallSet<StringRef, 12> Keywords;
  llvm::SmallSet<StringRef, 12> Literals;
};

} // namespace clang

#endif
