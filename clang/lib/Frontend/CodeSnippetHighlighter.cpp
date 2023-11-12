//===-- CodeSnippetHighlighter.cpp - Code snippet highlighting --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Frontend/CodeSnippetHighlighter.h"
#include "clang/Basic/CharInfo.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/Support/raw_ostream.h"
#include <chrono>

using namespace clang;

// Magenta is taken for 'warning'. Red is already 'error' and 'cyan'
// is already taken for 'note'. Green is already used to underline
// source ranges. White and black are bad because of the usual
// terminal backgrounds. Which leaves us only with TWO options.
static constexpr raw_ostream::Colors CommentColor = raw_ostream::YELLOW;
static constexpr raw_ostream::Colors LiteralColor = raw_ostream::GREEN;
static constexpr raw_ostream::Colors KeywordColor = raw_ostream::BLUE;
/// Maximum size of file we still highlight.
static constexpr size_t MaxBufferSize = 1024 * 1024; // 1MB.

std::unique_ptr<llvm::SmallVector<StyleRange>[]>
CodeSnippetHighlighter::highlightLines(unsigned StartLineNumber,
                                       unsigned EndLineNumber,
                                       const Preprocessor *PP,
                                       const LangOptions &LangOpts, FileID FID,
                                       const SourceManager &SM) {
  assert(StartLineNumber <= EndLineNumber);
  auto SnippetRanges = std::make_unique<llvm::SmallVector<StyleRange>[]>(
      EndLineNumber - StartLineNumber + 1);

  if (!PP)
    return SnippetRanges;

  // Might cause emission of another diagnostic.
  if (PP->getIdentifierTable().getExternalIdentifierLookup())
    return SnippetRanges;

  auto Buff = SM.getBufferOrNone(FID);
  if (!Buff || Buff->getBufferSize() > MaxBufferSize)
    return SnippetRanges;

  Lexer L{FID, *Buff, SM, LangOpts};
  L.SetKeepWhitespaceMode(true);

  // Classify the given token and append it to the given vector.
  auto appendStyle = [PP, &LangOpts](llvm::SmallVector<StyleRange> &Vec,
                                     const Token &T, unsigned Start,
                                     unsigned Length) -> void {
    if (T.is(tok::raw_identifier)) {
      StringRef RawIdent = T.getRawIdentifier();
      // Special case true/false/nullptr literals, since they will otherwise be
      // treated as keywords.
      if (RawIdent == "true" || RawIdent == "false" || RawIdent == "nullptr") {
        Vec.emplace_back(Start, Start + Length, LiteralColor);
      } else {
        const IdentifierInfo *II = PP->getIdentifierInfo(RawIdent);
        assert(II);
        if (II->isKeyword(LangOpts))
          Vec.emplace_back(Start, Start + Length, KeywordColor);
      }
    } else if (tok::isLiteral(T.getKind())) {
      Vec.emplace_back(Start, Start + Length, LiteralColor);
    } else {
      assert(T.is(tok::comment));
      Vec.emplace_back(Start, Start + Length, CommentColor);
    }
  };


  bool Stop = false;
  while (!Stop) {
    Token T;
    Stop = L.LexFromRawLexer(T);
    if (T.is(tok::unknown))
      continue;

    // We are only interested in identifiers, literals and comments.
    if (!T.is(tok::raw_identifier) && !T.is(tok::comment) &&
        !tok::isLiteral(T.getKind()))
      continue;

    bool Invalid = false;
    unsigned TokenEndLine = SM.getSpellingLineNumber(T.getEndLoc(), &Invalid);
    if (Invalid || TokenEndLine < StartLineNumber)
      continue;

    assert(TokenEndLine >= StartLineNumber);

    unsigned TokenStartLine =
        SM.getSpellingLineNumber(T.getLocation(), &Invalid);
    if (Invalid)
      continue;
    // If this happens, we're done.
    if (TokenStartLine > EndLineNumber)
      break;

    unsigned StartCol =
        SM.getSpellingColumnNumber(T.getLocation(), &Invalid) - 1;
    if (Invalid)
      continue;

    // Simple tokens.
    if (TokenStartLine == TokenEndLine) {
      llvm::SmallVector<StyleRange> &LineRanges =
          SnippetRanges[TokenStartLine - StartLineNumber];
      appendStyle(LineRanges, T, StartCol, T.getLength());
      continue;
    }
    assert((TokenEndLine - TokenStartLine) >= 1);

    // For tokens that span multiple lines (think multiline comments), we
    // divide them into multiple StyleRanges.
    unsigned EndCol = SM.getSpellingColumnNumber(T.getEndLoc(), &Invalid) - 1;
    if (Invalid)
      continue;

    std::string Spelling = Lexer::getSpelling(T, SM, LangOpts);

    unsigned L = TokenStartLine;
    unsigned LineLength = 0;
    for (unsigned I = 0; I <= Spelling.size(); ++I) {
      // This line is done.
      if (isVerticalWhitespace(Spelling[I]) || I == Spelling.size()) {
        llvm::SmallVector<StyleRange> &LineRanges =
            SnippetRanges[L - StartLineNumber];

        if (L == StartLineNumber) {
          if (L == TokenStartLine) // First line
            appendStyle(LineRanges, T, StartCol, LineLength);
          else if (L == TokenEndLine) // Last line
            appendStyle(LineRanges, T, 0, EndCol);
          else
            appendStyle(LineRanges, T, 0, LineLength);
        }

        ++L;
        if (L > EndLineNumber)
          break;
        LineLength = 0;
        continue;
      }
      ++LineLength;
    }
  }

  return SnippetRanges;
}
