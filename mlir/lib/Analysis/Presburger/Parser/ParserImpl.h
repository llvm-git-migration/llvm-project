//===- ParserImpl.h - Presburger Parser Implementation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_PRESBURGER_PARSER_PARSERIMPL_H
#define MLIR_ANALYSIS_PRESBURGER_PARSER_PARSERIMPL_H

#include "ParseStructs.h"
#include "ParserState.h"
#include "mlir/Support/LogicalResult.h"
#include <optional>
#include <variant>

namespace mlir::presburger {
template <typename T>
using function_ref = llvm::function_ref<T>;

/// These are the supported delimiters around operand lists and region
/// argument lists, used by parseOperandList.
enum class Delimiter {
  /// Zero or more operands with no delimiters.
  None,
  /// Parens surrounding zero or more operands.
  Paren,
  /// Square brackets surrounding zero or more operands.
  Square,
  /// <> brackets surrounding zero or more operands.
  LessGreater,
  /// {} brackets surrounding zero or more operands.
  Braces,
  /// Parens supporting zero or more operands, or nothing.
  OptionalParen,
  /// Square brackets supporting zero or more ops, or nothing.
  OptionalSquare,
  /// <> brackets supporting zero or more ops, or nothing.
  OptionalLessGreater,
  /// {} brackets surrounding zero or more operands, or nothing.
  OptionalBraces,
};

/// Lower precedence ops (all at the same precedence level). LNoOp is false in
/// the boolean sense.
enum AffineLowPrecOp {
  /// Null value.
  LNoOp,
  Add,
  Sub
};

/// Higher precedence ops - all at the same precedence level. HNoOp is false
/// in the boolean sense.
enum AffineHighPrecOp {
  /// Null value.
  HNoOp,
  Mul,
  FloorDiv,
  CeilDiv,
  Mod
};

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

/// This class implement support for parsing global entities like attributes and
/// types. It is intended to be subclassed by specialized subparsers that
/// include state.
class ParserImpl {
public:
  ParserImpl(ParserState &state) : state(state) {}

  // Helper methods to get stuff from the parser-global state.
  ParserState &getState() const { return state; }

  /// Parse a comma-separated list of elements up until the specified end token.
  ParseResult
  parseCommaSeparatedListUntil(Token::Kind rightToken,
                               function_ref<ParseResult()> parseElement,
                               bool allowEmptyList = true);

  /// Parse a list of comma-separated items with an optional delimiter.  If a
  /// delimiter is provided, then an empty list is allowed.  If not, then at
  /// least one element will be parsed.
  ParseResult
  parseCommaSeparatedList(Delimiter delimiter,
                          function_ref<ParseResult()> parseElementFn,
                          StringRef contextMessage = StringRef());

  /// Parse a comma separated list of elements that must have at least one entry
  /// in it.
  ParseResult
  parseCommaSeparatedList(function_ref<ParseResult()> parseElementFn) {
    return parseCommaSeparatedList(Delimiter::None, parseElementFn);
  }

  // We have two forms of parsing methods - those that return a non-null
  // pointer on success, and those that return a ParseResult to indicate whether
  // they returned a failure.  The second class fills in by-reference arguments
  // as the results of their action.

  //===--------------------------------------------------------------------===//
  // Error Handling
  //===--------------------------------------------------------------------===//

  /// Emit an error and return failure.
  ParseResult emitError(const Twine &message = {});
  ParseResult emitError(SMLoc loc, const Twine &message = {});

  /// Emit an error about a "wrong token".  If the current token is at the
  /// start of a source line, this will apply heuristics to back up and report
  /// the error at the end of the previous line, which is where the expected
  /// token is supposed to be.
  ParseResult emitWrongTokenError(const Twine &message = {});

  //===--------------------------------------------------------------------===//
  // Token Parsing
  //===--------------------------------------------------------------------===//

  /// Return the current token the parser is inspecting.
  const Token &getToken() const { return state.curToken; }
  StringRef getTokenSpelling() const { return state.curToken.getSpelling(); }

  /// Return the last parsed token.
  const Token &getLastToken() const { return state.lastToken; }

  /// If the current token has the specified kind, consume it and return true.
  /// If not, return false.
  bool consumeIf(Token::Kind kind) {
    if (state.curToken.isNot(kind))
      return false;
    consumeToken(kind);
    return true;
  }

  /// Advance the current lexer onto the next token.
  void consumeToken() {
    assert(state.curToken.isNot(Token::eof, Token::error) &&
           "shouldn't advance past EOF or errors");
    state.lastToken = state.curToken;
    state.curToken = state.lex.lexToken();
  }

  /// Advance the current lexer onto the next token, asserting what the expected
  /// current token is.  This is preferred to the above method because it leads
  /// to more self-documenting code with better checking.
  void consumeToken(Token::Kind kind) {
    assert(state.curToken.is(kind) && "consumed an unexpected token");
    consumeToken();
  }

  /// Reset the parser to the given lexer position.
  void resetToken(const char *tokPos) {
    state.lex.resetPointer(tokPos);
    state.lastToken = state.curToken;
    state.curToken = state.lex.lexToken();
  }

  /// Consume the specified token if present and return success.  On failure,
  /// output a diagnostic and return failure.
  ParseResult parseToken(Token::Kind expectedToken, const Twine &message);

  /// Parse an optional integer value from the stream.
  std::optional<ParseResult> parseOptionalInteger(APInt &result);

  /// Returns true if the current token corresponds to a keyword.
  bool isCurrentTokenAKeyword() const {
    return getToken().isAny(Token::bare_identifier, Token::inttype) ||
           getToken().isKeyword();
  }

  /// Parse a keyword, if present, into 'keyword'.
  ParseResult parseOptionalKeyword(StringRef *keyword);

  //===--------------------------------------------------------------------===//
  // Affine Parsing
  //===--------------------------------------------------------------------===//

  ParseResult
  parseAffineExprReference(ArrayRef<std::pair<StringRef, AffineExpr>> symbolSet,
                           AffineExpr &expr);
  ParseResult
  parseAffineExprInline(ArrayRef<std::pair<StringRef, AffineExpr>> symbolSet,
                        AffineExpr &expr);
  std::optional<AffineMap> parseAffineMapRange(unsigned numDims,
                                               unsigned numSymbols);
  std::optional<IntegerSet> parseIntegerSetConstraints(unsigned numDims,
                                                       unsigned numSymbols);
  std::variant<AffineMap, IntegerSet, std::nullopt_t>
  parseAffineMapOrIntegerSet();

private:
  // Binary affine op parsing.
  AffineLowPrecOp consumeIfLowPrecOp();
  AffineHighPrecOp consumeIfHighPrecOp();

  // Identifier lists for polyhedral structures.
  ParseResult parseDimIdList(unsigned &numDims);
  ParseResult parseSymbolIdList(unsigned &numSymbols);
  ParseResult parseDimAndOptionalSymbolIdList(unsigned &numDims,
                                              unsigned &numSymbols);
  ParseResult parseIdentifierDefinition(
      std::variant<AffineDimExpr, AffineSymbolExpr> idExpr);

  AffineExpr parseAffineExpr();
  AffineExpr parseParentheticalExpr();
  AffineExpr parseNegateExpression(const AffineExpr &lhs);
  AffineExpr parseIntegerExpr();
  AffineExpr parseBareIdExpr();

  AffineExpr getAffineBinaryOpExpr(AffineHighPrecOp op, AffineExpr &&lhs,
                                   AffineExpr &&rhs, SMLoc opLoc);
  AffineExpr getAffineBinaryOpExpr(AffineLowPrecOp op, AffineExpr &&lhs,
                                   AffineExpr &&rhs);
  AffineExpr parseAffineOperandExpr(const AffineExpr &lhs);
  AffineExpr parseAffineLowPrecOpExpr(AffineExpr &&llhs,
                                      AffineLowPrecOp llhsOp);
  AffineExpr parseAffineHighPrecOpExpr(AffineExpr &&llhs,
                                       AffineHighPrecOp llhsOp,
                                       SMLoc llhsOpLoc);
  AffineExpr parseAffineConstraint(bool *isEq);

private:
  ParserState &state;
  function_ref<ParseResult(bool)> parseElement;
  unsigned numDimOperands = 0;
  unsigned numSymbolOperands = 0;
  SmallVector<
      std::pair<StringRef, std::variant<AffineDimExpr, AffineSymbolExpr>>, 4>
      dimsAndSymbols;
};
} // namespace mlir::presburger

#endif // MLIR_ANALYSIS_PRESBURGER_PARSER_PARSERIMPL_H
