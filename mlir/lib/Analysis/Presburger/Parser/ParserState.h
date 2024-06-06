//===- ParserState.h - MLIR Presburger ParserState --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PARSER_PARSERSTATE_H
#define MLIR_ANALYSIS_PRESBURGER_PARSER_PARSERSTATE_H

#include "Lexer.h"
#include "llvm/Support/SourceMgr.h"

namespace mlir {
namespace presburger {
namespace detail {

//===----------------------------------------------------------------------===//
// ParserState
//===----------------------------------------------------------------------===//

/// This class refers to all of the state maintained globally by the parser,
/// such as the current lexer position etc.
struct ParserState {
  ParserState(const llvm::SourceMgr &sourceMgr)
      : sourceMgr(sourceMgr), lex(sourceMgr), curToken(lex.lexToken()),
        lastToken(Token::error, "") {}
  ParserState(const ParserState &) = delete;
  void operator=(const ParserState &) = delete;

  // The source manager for the parser.
  const llvm::SourceMgr &sourceMgr;

  /// The lexer for the source file we're parsing.
  Lexer lex;

  /// This is the next token that hasn't been consumed yet.
  Token curToken;

  /// This is the last token that has been consumed.
  Token lastToken;
};
} // namespace detail
} // namespace presburger
} // namespace mlir

#endif // MLIR_ANALYSIS_PRESBURGER_PARSER_PARSERSTATE_H
