//===- ParserImpl.cpp - Presburger Parser Implementation --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the ParserImpl class for the Presburger textual form.
//
//===----------------------------------------------------------------------===//

#include "ParserImpl.h"
#include "Flattener.h"
#include "ParseStructs.h"
#include "ParserState.h"
#include "mlir/Analysis/Presburger/Parser.h"

namespace mlir {
namespace presburger {
namespace detail {
using llvm::MemoryBuffer;
using llvm::SmallVector;
using llvm::SourceMgr;

//===----------------------------------------------------------------------===//
// Parser core
//===----------------------------------------------------------------------===//
/// Consume the specified token if present and return success.  On failure,
/// output a diagnostic and return failure.
ParseResult ParserImpl::parseToken(Token::Kind expectedToken,
                                   const Twine &message) {
  if (consumeIf(expectedToken))
    return success();
  return emitWrongTokenError(message);
}

/// Parse a list of comma-separated items with an optional delimiter.  If a
/// delimiter is provided, then an empty list is allowed.  If not, then at
/// least one element will be parsed.
ParseResult
ParserImpl::parseCommaSeparatedList(Delimiter delimiter,
                                    function_ref<ParseResult()> parseElementFn,
                                    StringRef contextMessage) {
  switch (delimiter) {
  case Delimiter::None:
    break;
  case Delimiter::OptionalParen:
    if (getToken().isNot(Token::l_paren))
      return success();
    [[fallthrough]];
  case Delimiter::Paren:
    if (parseToken(Token::l_paren, "expected '('" + contextMessage))
      return failure();
    // Check for empty list.
    if (consumeIf(Token::r_paren))
      return success();
    break;
  case Delimiter::OptionalLessGreater:
    // Check for absent list.
    if (getToken().isNot(Token::less))
      return success();
    [[fallthrough]];
  case Delimiter::LessGreater:
    if (parseToken(Token::less, "expected '<'" + contextMessage))
      return success();
    // Check for empty list.
    if (consumeIf(Token::greater))
      return success();
    break;
  case Delimiter::OptionalSquare:
    if (getToken().isNot(Token::l_square))
      return success();
    [[fallthrough]];
  case Delimiter::Square:
    if (parseToken(Token::l_square, "expected '['" + contextMessage))
      return failure();
    // Check for empty list.
    if (consumeIf(Token::r_square))
      return success();
    break;
  case Delimiter::OptionalBraces:
    if (getToken().isNot(Token::l_brace))
      return success();
    [[fallthrough]];
  case Delimiter::Braces:
    if (parseToken(Token::l_brace, "expected '{'" + contextMessage))
      return failure();
    // Check for empty list.
    if (consumeIf(Token::r_brace))
      return success();
    break;
  }

  // Non-empty case starts with an element.
  if (parseElementFn())
    return failure();

  // Otherwise we have a list of comma separated elements.
  while (consumeIf(Token::comma)) {
    if (parseElementFn())
      return failure();
  }

  switch (delimiter) {
  case Delimiter::None:
    return success();
  case Delimiter::OptionalParen:
  case Delimiter::Paren:
    return parseToken(Token::r_paren, "expected ')'" + contextMessage);
  case Delimiter::OptionalLessGreater:
  case Delimiter::LessGreater:
    return parseToken(Token::greater, "expected '>'" + contextMessage);
  case Delimiter::OptionalSquare:
  case Delimiter::Square:
    return parseToken(Token::r_square, "expected ']'" + contextMessage);
  case Delimiter::OptionalBraces:
  case Delimiter::Braces:
    return parseToken(Token::r_brace, "expected '}'" + contextMessage);
  }
  llvm_unreachable("Unknown delimiter");
}

//===----------------------------------------------------------------------===//
// Parse error emitters
//===----------------------------------------------------------------------===//
ParseResult ParserImpl::emitError(SMLoc loc, const Twine &message) {
  // If we hit a parse error in response to a lexer error, then the lexer
  // already reported the error.
  if (!getToken().is(Token::error))
    state.sourceMgr.PrintMessage(loc, SourceMgr::DK_Error, message);
  return failure();
}

ParseResult ParserImpl::emitError(const Twine &message) {
  SMLoc loc = state.curToken.getLoc();
  if (state.curToken.isNot(Token::eof))
    return emitError(loc, message);

  // If the error is to be emitted at EOF, move it back one character.
  return emitError(SMLoc::getFromPointer(loc.getPointer() - 1), message);
}

/// Emit an error about a "wrong token".  If the current token is at the
/// start of a source line, this will apply heuristics to back up and report
/// the error at the end of the previous line, which is where the expected
/// token is supposed to be.
ParseResult ParserImpl::emitWrongTokenError(const Twine &message) {
  SMLoc loc = state.curToken.getLoc();

  // If the error is to be emitted at EOF, move it back one character.
  if (state.curToken.is(Token::eof))
    loc = SMLoc::getFromPointer(loc.getPointer() - 1);

  // This is the location we were originally asked to report the error at.
  SMLoc originalLoc = loc;

  // Determine if the token is at the start of the current line.
  const char *bufferStart = state.lex.getBufferBegin();
  const char *curPtr = loc.getPointer();

  // Use this StringRef to keep track of what we are going to back up through,
  // it provides nicer string search functions etc.
  StringRef startOfBuffer(bufferStart, curPtr - bufferStart);

  // Back up over entirely blank lines.
  while (true) {
    // Back up until we see a \n, but don't look past the buffer start.
    startOfBuffer = startOfBuffer.rtrim(" \t");

    // For tokens with no preceding source line, just emit at the original
    // location.
    if (startOfBuffer.empty())
      return emitError(originalLoc, message);

    // If we found something that isn't the end of line, then we're done.
    if (startOfBuffer.back() != '\n' && startOfBuffer.back() != '\r')
      return emitError(SMLoc::getFromPointer(startOfBuffer.end()), message);

    // Drop the \n so we emit the diagnostic at the end of the line.
    startOfBuffer = startOfBuffer.drop_back();
  }
}

//===----------------------------------------------------------------------===//
// Affine Expression Parser
//===----------------------------------------------------------------------===//
static bool isIdentifier(const Token &token) {
  // We include only `inttype` and `bare_identifier` here since they are the
  // only non-keyword tokens that can be used to represent an identifier.
  return token.isAny(Token::bare_identifier, Token::inttype) ||
         token.isKeyword();
}

/// Parse a bare id that may appear in an affine expression.
///
///   affine-expr ::= bare-id
AffineExpr ParserImpl::parseBareIdExpr() {
  if (!isIdentifier(getToken())) {
    std::ignore = emitWrongTokenError("expected bare identifier");
    return nullptr;
  }

  StringRef sRef = getTokenSpelling();
  for (const auto &entry : dimsAndSymbols) {
    if (entry.first == sRef) {
      consumeToken();
      // Since every DimExpr or SymbolExpr is used more than once, construct a
      // fresh unique_ptr every time we encounter it in the dimsAndSymbols list.
      if (std::holds_alternative<AffineDimExpr>(entry.second))
        return std::make_unique<AffineDimExpr>(
            std::get<AffineDimExpr>(entry.second));
      return std::make_unique<AffineSymbolExpr>(
          std::get<AffineSymbolExpr>(entry.second));
    }
  }

  std::ignore = emitWrongTokenError("use of undeclared identifier");
  return nullptr;
}

/// Parse an affine expression inside parentheses.
///
///   affine-expr ::= `(` affine-expr `)`
AffineExpr ParserImpl::parseParentheticalExpr() {
  if (parseToken(Token::l_paren, "expected '('"))
    return nullptr;
  if (getToken().is(Token::r_paren)) {
    std::ignore = emitError("no expression inside parentheses");
    return nullptr;
  }

  AffineExpr expr = parseAffineExpr();
  if (!expr || parseToken(Token::r_paren, "expected ')'"))
    return nullptr;

  return expr;
}

/// Parse the negation expression.
///
///   affine-expr ::= `-` affine-expr
AffineExpr ParserImpl::parseNegateExpression(const AffineExpr &lhs) {
  if (parseToken(Token::minus, "expected '-'"))
    return nullptr;

  AffineExpr operand = parseAffineOperandExpr(lhs);
  // Since negation has the highest precedence of all ops (including high
  // precedence ops) but lower than parentheses, we are only going to use
  // parseAffineOperandExpr instead of parseAffineExpr here.
  if (!operand) {
    // Extra error message although parseAffineOperandExpr would have
    // complained. Leads to a better diagnostic.
    std::ignore = emitError("missing operand of negation");
    return nullptr;
  }
  return -1 * std::move(operand);
}

/// Parse a positive integral constant appearing in an affine expression.
///
///   affine-expr ::= integer-literal
AffineExpr ParserImpl::parseIntegerExpr() {
  std::optional<uint64_t> val = getToken().getUInt64IntegerValue();
  if (!val.has_value() || (int64_t)*val < 0) {
    std::ignore = emitError("constant too large for index");
    return nullptr;
  }

  consumeToken(Token::integer);
  return std::make_unique<AffineConstantExpr>((int64_t)*val);
}

/// Parses an expression that can be a valid operand of an affine expression.
/// lhs: if non-null, lhs is an affine expression that is the lhs of a binary
/// operator, the rhs of which is being parsed. This is used to determine
/// whether an error should be emitted for a missing right operand.
//  Eg: for an expression without parentheses (like i + j + k + l), each
//  of the four identifiers is an operand. For i + j*k + l, j*k is not an
//  operand expression, it's an op expression and will be parsed via
//  parseAffineHighPrecOpExpression(). However, for i + (j*k) + -l, (j*k) and
//  -l are valid operands that will be parsed by this function.
AffineExpr ParserImpl::parseAffineOperandExpr(const AffineExpr &lhs) {
  switch (getToken().getKind()) {
  case Token::integer:
    return parseIntegerExpr();
  case Token::l_paren:
    return parseParentheticalExpr();
  case Token::minus:
    return parseNegateExpression(lhs);
  case Token::kw_ceildiv:
  case Token::kw_floordiv:
  case Token::kw_mod:
    // Try to treat these tokens as identifiers.
    return parseBareIdExpr();
  case Token::plus:
  case Token::star:
    if (lhs)
      std::ignore = emitError("missing right operand of binary operator");
    else
      std::ignore = emitError("missing left operand of binary operator");
    return nullptr;
  default:
    // If nothing matches, we try to treat this token as an identifier.
    if (isIdentifier(getToken()))
      return parseBareIdExpr();

    if (lhs)
      std::ignore = emitError("missing right operand of binary operator");
    else
      std::ignore = emitError("expected affine expression");
    return nullptr;
  }
}

/// Create an affine binary high precedence op expression (mul's, div's, mod).
/// opLoc is the location of the op token to be used to report errors
/// for non-conforming expressions.
AffineExpr ParserImpl::getAffineBinaryOpExpr(AffineHighPrecOp op,
                                             AffineExpr &&lhs, AffineExpr &&rhs,
                                             SMLoc opLoc) {
  switch (op) {
  case Mul:
    if (!lhs->isSymbolicOrConstant() && !rhs->isSymbolicOrConstant()) {
      std::ignore = emitError(
          opLoc, "non-affine expression: at least one of the multiply "
                 "operands has to be either a constant or symbolic");
      return nullptr;
    }
    return std::move(lhs) * std::move(rhs);
  case FloorDiv:
    if (!rhs->isSymbolicOrConstant()) {
      std::ignore =
          emitError(opLoc, "non-affine expression: right operand of floordiv "
                           "has to be either a constant or symbolic");
      return nullptr;
    }
    return floorDiv(std::move(lhs), std::move(rhs));
  case CeilDiv:
    if (!rhs->isSymbolicOrConstant()) {
      std::ignore =
          emitError(opLoc, "non-affine expression: right operand of ceildiv "
                           "has to be either a constant or symbolic");
      return nullptr;
    }
    return ceilDiv(std::move(lhs), std::move(rhs));
  case Mod:
    if (!rhs->isSymbolicOrConstant()) {
      std::ignore =
          emitError(opLoc, "non-affine expression: right operand of mod "
                           "has to be either a constant or symbolic");
      return nullptr;
    }
    return std::move(lhs) % std::move(rhs);
  case HNoOp:
    llvm_unreachable("can't create affine expression for null high prec op");
    return nullptr;
  }
  llvm_unreachable("Unknown AffineHighPrecOp");
}

/// Create an affine binary low precedence op expression (add, sub).
AffineExpr ParserImpl::getAffineBinaryOpExpr(AffineLowPrecOp op,
                                             AffineExpr &&lhs,
                                             AffineExpr &&rhs) {
  switch (op) {
  case AffineLowPrecOp::Add:
    return std::move(lhs) + std::move(rhs);
  case AffineLowPrecOp::Sub:
    return std::move(lhs) - std::move(rhs);
  case AffineLowPrecOp::LNoOp:
    llvm_unreachable("can't create affine expression for null low prec op");
    return nullptr;
  }
  llvm_unreachable("Unknown AffineLowPrecOp");
}

/// Consume this token if it is a lower precedence affine op (there are only
/// two precedence levels).
AffineLowPrecOp ParserImpl::consumeIfLowPrecOp() {
  switch (getToken().getKind()) {
  case Token::plus:
    consumeToken(Token::plus);
    return AffineLowPrecOp::Add;
  case Token::minus:
    consumeToken(Token::minus);
    return AffineLowPrecOp::Sub;
  default:
    return AffineLowPrecOp::LNoOp;
  }
}

/// Consume this token if it is a higher precedence affine op (there are only
/// two precedence levels)
AffineHighPrecOp ParserImpl::consumeIfHighPrecOp() {
  switch (getToken().getKind()) {
  case Token::star:
    consumeToken(Token::star);
    return Mul;
  case Token::kw_floordiv:
    consumeToken(Token::kw_floordiv);
    return FloorDiv;
  case Token::kw_ceildiv:
    consumeToken(Token::kw_ceildiv);
    return CeilDiv;
  case Token::kw_mod:
    consumeToken(Token::kw_mod);
    return Mod;
  default:
    return HNoOp;
  }
}

/// Parse a high precedence op expression list: mul, div, and mod are high
/// precedence binary ops, i.e., parse a
///   expr_1 op_1 expr_2 op_2 ... expr_n
/// where op_1, op_2 are all a AffineHighPrecOp (mul, div, mod).
/// All affine binary ops are left associative.
/// Given llhs, returns (llhs llhsOp lhs) op rhs, or (lhs op rhs) if llhs is
/// null. If no rhs can be found, returns (llhs llhsOp lhs) or lhs if llhs is
/// null. llhsOpLoc is the location of the llhsOp token that will be used to
/// report an error for non-conforming expressions.
AffineExpr ParserImpl::parseAffineHighPrecOpExpr(AffineExpr &&llhs,
                                                 AffineHighPrecOp llhsOp,
                                                 SMLoc llhsOpLoc) {
  AffineExpr lhs = parseAffineOperandExpr(llhs);
  if (!lhs)
    return nullptr;

  // Found an LHS. Parse the remaining expression.
  SMLoc opLoc = getToken().getLoc();
  if (AffineHighPrecOp op = consumeIfHighPrecOp()) {
    if (llhs) {
      AffineExpr expr =
          getAffineBinaryOpExpr(llhsOp, std::move(llhs), std::move(lhs), opLoc);
      if (!expr)
        return nullptr;
      return parseAffineHighPrecOpExpr(std::move(expr), op, opLoc);
    }
    // No LLHS, get RHS
    return parseAffineHighPrecOpExpr(std::move(lhs), op, opLoc);
  }

  // This is the last operand in this expression.
  if (llhs)
    return getAffineBinaryOpExpr(llhsOp, std::move(llhs), std::move(lhs),
                                 llhsOpLoc);

  // No llhs, 'lhs' itself is the expression.
  return lhs;
}

/// Parse affine expressions that are bare-id's, integer constants,
/// parenthetical affine expressions, and affine op expressions that are a
/// composition of those.
///
/// All binary op's associate from left to right.
///
/// {add, sub} have lower precedence than {mul, div, and mod}.
///
/// Add, sub'are themselves at the same precedence level. Mul, floordiv,
/// ceildiv, and mod are at the same higher precedence level. Negation has
/// higher precedence than any binary op.
///
/// llhs: the affine expression appearing on the left of the one being parsed.
/// This function will return ((llhs llhsOp lhs) op rhs) if llhs is non null,
/// and lhs op rhs otherwise; if there is no rhs, llhs llhsOp lhs is returned
/// if llhs is non-null; otherwise lhs is returned. This is to deal with left
/// associativity.
///
/// Eg: when the expression is e1 + e2*e3 + e4, with e1 as llhs, this function
/// will return the affine expr equivalent of (e1 + (e2*e3)) + e4, where
/// (e2*e3) will be parsed using parseAffineHighPrecOpExpr().
AffineExpr ParserImpl::parseAffineLowPrecOpExpr(AffineExpr &&llhs,
                                                AffineLowPrecOp llhsOp) {
  AffineExpr lhs = parseAffineOperandExpr(llhs);
  if (!lhs)
    return nullptr;

  // Found an LHS. Deal with the ops.
  if (AffineLowPrecOp lOp = consumeIfLowPrecOp()) {
    if (llhs) {
      AffineExpr sum =
          getAffineBinaryOpExpr(llhsOp, std::move(llhs), std::move(lhs));
      return parseAffineLowPrecOpExpr(std::move(sum), lOp);
    }
    // No LLHS, get RHS and form the expression.
    return parseAffineLowPrecOpExpr(std::move(lhs), lOp);
  }
  SMLoc opLoc = getToken().getLoc();
  if (AffineHighPrecOp hOp = consumeIfHighPrecOp()) {
    // We have a higher precedence op here. Get the rhs operand for the llhs
    // through parseAffineHighPrecOpExpr.
    AffineExpr highRes = parseAffineHighPrecOpExpr(std::move(lhs), hOp, opLoc);
    if (!highRes)
      return nullptr;

    // If llhs is null, the product forms the first operand of the yet to be
    // found expression. If non-null, the op to associate with llhs is llhsOp.
    AffineExpr expr = llhs ? getAffineBinaryOpExpr(llhsOp, std::move(llhs),
                                                   std::move(highRes))
                           : std::move(highRes);

    // Recurse for subsequent low prec op's after the affine high prec op
    // expression.
    if (AffineLowPrecOp nextOp = consumeIfLowPrecOp())
      return parseAffineLowPrecOpExpr(std::move(expr), nextOp);
    return expr;
  }
  // Last operand in the expression list.
  if (llhs)
    return getAffineBinaryOpExpr(llhsOp, std::move(llhs), std::move(lhs));
  // No llhs, 'lhs' itself is the expression.
  return lhs;
}

/// Parse an affine expression.
///  affine-expr ::= `(` affine-expr `)`
///                | `-` affine-expr
///                | affine-expr `+` affine-expr
///                | affine-expr `-` affine-expr
///                | affine-expr `*` affine-expr
///                | affine-expr `floordiv` affine-expr
///                | affine-expr `ceildiv` affine-expr
///                | affine-expr `mod` affine-expr
///                | bare-id
///                | integer-literal
///
/// Additional conditions are checked depending on the production. For eg.,
/// one of the operands for `*` has to be either constant/symbolic; the second
/// operand for floordiv, ceildiv, and mod has to be a positive integer.
AffineExpr ParserImpl::parseAffineExpr() {
  return parseAffineLowPrecOpExpr(nullptr, AffineLowPrecOp::LNoOp);
}

/// Parse a dim or symbol from the lists appearing before the actual
/// expressions of the affine map. Update our state to store the
/// dimensional/symbolic identifier.
ParseResult ParserImpl::parseIdentifierDefinition(
    std::variant<AffineDimExpr, AffineSymbolExpr> idExpr) {
  if (!isIdentifier(getToken()))
    return emitWrongTokenError("expected bare identifier");

  StringRef name = getTokenSpelling();
  for (const auto &entry : dimsAndSymbols) {
    if (entry.first == name)
      return emitError("redefinition of identifier '" + name + "'");
  }
  consumeToken();

  dimsAndSymbols.emplace_back(name, idExpr);
  return success();
}

/// Parse the list of dimensional identifiers to an affine map.
ParseResult ParserImpl::parseDimIdList(unsigned &numDims) {
  auto parseElt = [&]() -> ParseResult {
    return parseIdentifierDefinition(AffineDimExpr(numDims++));
  };
  return parseCommaSeparatedList(Delimiter::Paren, parseElt,
                                 " in dimensional identifier list");
}

/// Parse the list of symbolic identifiers to an affine map.
ParseResult ParserImpl::parseSymbolIdList(unsigned &numSymbols) {
  auto parseElt = [&]() -> ParseResult {
    return parseIdentifierDefinition(AffineSymbolExpr(numSymbols++));
  };
  return parseCommaSeparatedList(Delimiter::Square, parseElt,
                                 " in symbol list");
}

/// Parse the list of symbolic identifiers to an affine map.
ParseResult ParserImpl::parseDimAndOptionalSymbolIdList(unsigned &numDims,
                                                        unsigned &numSymbols) {
  if (parseDimIdList(numDims)) {
    return failure();
  }
  if (!getToken().is(Token::l_square)) {
    numSymbols = 0;
    return success();
  }
  return parseSymbolIdList(numSymbols);
}

/// Parse the range and sizes affine map definition inline.
///
///  affine-map ::= dim-and-symbol-id-lists `->` multi-dim-affine-expr
///
///  multi-dim-affine-expr ::= `(` `)`
///  multi-dim-affine-expr ::= `(` affine-expr (`,` affine-expr)* `)`
std::optional<AffineMap> ParserImpl::parseAffineMapRange(unsigned numDims,
                                                         unsigned numSymbols) {
  SmallVector<AffineExpr, 4> exprs;
  auto parseElt = [&]() -> ParseResult {
    AffineExpr elt = parseAffineExpr();
    ParseResult res = elt ? success() : failure();
    exprs.emplace_back(std::move(elt));
    return res;
  };

  // Parse a multi-dimensional affine expression (a comma-separated list of
  // 1-d affine expressions). Grammar:
  // multi-dim-affine-expr ::= `(` `)`
  //                         | `(` affine-expr (`,` affine-expr)* `)`
  if (parseCommaSeparatedList(Delimiter::Paren, parseElt,
                              " in affine map range"))
    return std::nullopt;

  // Parsed a valid affine map.
  return AffineMap(numDims, numSymbols, std::move(exprs));
}

/// Parse an affine constraint.
///  affine-constraint ::= affine-expr `>=` `affine-expr`
///                      | affine-expr `<=` `affine-expr`
///                      | affine-expr `==` `affine-expr`
///
/// The constraint is normalized to
///  affine-constraint ::= affine-expr `>=` `0`
///                      | affine-expr `==` `0`
/// before returning.
///
/// isEq is set to true if the parsed constraint is an equality, false if it
/// is an inequality (greater than or equal).
///
AffineExpr ParserImpl::parseAffineConstraint(bool *isEq) {
  AffineExpr lhsExpr = parseAffineExpr();
  if (!lhsExpr)
    return nullptr;

  // affine-constraint ::= `affine-expr` `>=` `affine-expr`
  if (consumeIf(Token::greater) && consumeIf(Token::equal)) {
    AffineExpr rhsExpr = parseAffineExpr();
    if (!rhsExpr)
      return nullptr;
    *isEq = false;
    return std::move(lhsExpr) - std::move(rhsExpr);
  }

  // affine-constraint ::= `affine-expr` `<=` `affine-expr`
  if (consumeIf(Token::less) && consumeIf(Token::equal)) {
    AffineExpr rhsExpr = parseAffineExpr();
    if (!rhsExpr)
      return nullptr;
    *isEq = false;
    return std::move(rhsExpr) - std::move(lhsExpr);
  }

  // affine-constraint ::= `affine-expr` `==` `affine-expr`
  if (consumeIf(Token::equal) && consumeIf(Token::equal)) {
    AffineExpr rhsExpr = parseAffineExpr();
    if (!rhsExpr)
      return nullptr;
    *isEq = true;
    return std::move(lhsExpr) - std::move(rhsExpr);
  }

  std::ignore =
      emitError("expected '== affine-expr' or '>= affine-expr' at end of "
                "affine constraint");
  return nullptr;
}

/// Parse the constraints that are part of an integer set definition.
///  integer-set-inline
///                ::= dim-and-symbol-id-lists `:`
///                '(' affine-constraint-conjunction? ')'
///  affine-constraint-conjunction ::= affine-constraint (`,`
///                                       affine-constraint)*
///
std::optional<IntegerSet>
ParserImpl::parseIntegerSetConstraints(unsigned numDims, unsigned numSymbols) {
  SmallVector<AffineExpr, 4> constraints;
  SmallVector<bool, 4> isEqs;
  auto parseElt = [&]() -> ParseResult {
    bool isEq;
    AffineExpr elt = parseAffineConstraint(&isEq);
    ParseResult res = elt ? success() : failure();
    if (elt) {
      constraints.emplace_back(std::move(elt));
      isEqs.push_back(isEq);
    }
    return res;
  };

  // Parse a list of affine constraints (comma-separated).
  if (parseCommaSeparatedList(Delimiter::Paren, parseElt,
                              " in integer set constraint list"))
    return std::nullopt;

  // If no constraints were parsed, then treat this as a degenerate 'true' case.
  if (constraints.empty()) {
    /* 0 == 0 */
    return IntegerSet(numDims, numSymbols,
                      std::make_unique<AffineConstantExpr>(0), true);
  }

  // Parsed a valid integer set.
  return IntegerSet(numDims, numSymbols, std::move(constraints),
                    std::move(isEqs));
}

std::variant<AffineMap, IntegerSet, std::nullopt_t>
ParserImpl::parseAffineMapOrIntegerSet() {
  unsigned numDims = 0, numSymbols = 0;

  // List of dimensional and optional symbol identifiers.
  if (parseDimAndOptionalSymbolIdList(numDims, numSymbols))
    return std::nullopt;

  if (consumeIf(Token::arrow)) {
    if (std::optional<AffineMap> v = parseAffineMapRange(numDims, numSymbols))
      return std::move(*v);
    return std::nullopt;
  }

  if (parseToken(Token::colon, "expected '->' or ':'"))
    return std::nullopt;

  if (std::optional<IntegerSet> v =
          parseIntegerSetConstraints(numDims, numSymbols))
    return std::move(*v);
  return std::nullopt;
}

static MultiAffineFunction getMultiAffineFunctionFromMap(const AffineMap &map) {
  IntegerPolyhedron cst(presburger::PresburgerSpace::getSetSpace(0, 0, 0));
  std::vector<SmallVector<int64_t, 8>> flattenedExprs;

  // Flatten expressions and add them to the constraint system.
  LogicalResult result = getFlattenedAffineExprs(map, flattenedExprs, cst);
  assert(result.succeeded() && "Unable to get flattened affine exprs");

  DivisionRepr divs = cst.getLocalReprs();
  assert(divs.hasAllReprs() &&
         "AffineMap cannot produce divs without local representation");

  // TODO: We shouldn't have to do this conversion.
  Matrix<MPInt> mat(map.getNumExprs(),
                    map.getNumInputs() + divs.getNumDivs() + 1);
  for (unsigned i = 0; i < flattenedExprs.size(); ++i)
    for (unsigned j = 0; j < flattenedExprs[i].size(); ++j)
      mat(i, j) = flattenedExprs[i][j];

  return MultiAffineFunction(
      PresburgerSpace::getRelationSpace(map.getNumDims(), map.getNumExprs(),
                                        map.getNumSymbols(), divs.getNumDivs()),
      mat, divs);
}

static IntegerPolyhedron getPolyhedronFromSet(const IntegerSet &set) {
  IntegerPolyhedron cst(presburger::PresburgerSpace::getSetSpace(0, 0, 0));
  std::vector<SmallVector<int64_t, 8>> flattenedExprs;

  // Flatten expressions and add them to the constraint system.
  LogicalResult result = getFlattenedAffineExprs(set, flattenedExprs, cst);
  assert(result.succeeded() && "Unable to get flattened affine exprs");
  assert(flattenedExprs.size() == set.getNumConstraints());

  unsigned numInequalities = set.getNumInequalities();
  unsigned numEqualities = set.getNumEqualities();
  unsigned numDims = set.getNumDims();
  unsigned numSymbols = set.getNumSymbols();
  unsigned numReservedCols = numDims + numSymbols + 1;
  IntegerPolyhedron poly(
      numInequalities, numEqualities, numReservedCols,
      presburger::PresburgerSpace::getSetSpace(numDims, numSymbols, 0));
  assert(numReservedCols >= poly.getSpace().getNumVars() + 1);

  poly.insertVar(VarKind::Local, poly.getNumVarKind(VarKind::Local),
                 /*num=*/cst.getNumLocalVars());

  for (unsigned i = 0; i < flattenedExprs.size(); ++i) {
    const auto &flatExpr = flattenedExprs[i];
    assert(flatExpr.size() == poly.getSpace().getNumVars() + 1);
    if (set.eqFlags[i])
      poly.addEquality(flatExpr);
    else
      poly.addInequality(flatExpr);
  }
  // Add the other constraints involving local vars from flattening.
  poly.append(cst);

  return poly;
}

static std::variant<AffineMap, IntegerSet, std::nullopt_t>
parseAffineMapOrIntegerSet(StringRef inputStr) {
  SourceMgr sourceMgr;
  auto memBuffer = MemoryBuffer::getMemBuffer(
      inputStr, /*BufferName=*/"<mlir_parser_buffer>",
      /*RequiresNullTerminator=*/false);
  sourceMgr.AddNewSourceBuffer(std::move(memBuffer), SMLoc());
  ParserState state(sourceMgr);
  ParserImpl parser(state);
  return parser.parseAffineMapOrIntegerSet();
}

static AffineMap parseAffineMap(StringRef inputStr) {
  std::variant<AffineMap, IntegerSet, std::nullopt_t> v =
      detail::parseAffineMapOrIntegerSet(inputStr);
  if (std::holds_alternative<AffineMap>(v))
    return std::move(std::get<AffineMap>(v));
  llvm_unreachable("expected string to represent AffineMap");
}

static IntegerSet parseIntegerSet(StringRef inputStr) {
  std::variant<AffineMap, IntegerSet, std::nullopt_t> v =
      detail::parseAffineMapOrIntegerSet(inputStr);
  if (std::holds_alternative<IntegerSet>(v))
    return std::move(std::get<IntegerSet>(v));
  llvm_unreachable("expected string to represent IntegerSet");
}
} // namespace detail

IntegerPolyhedron parseIntegerPolyhedron(StringRef inputStr) {
  return detail::getPolyhedronFromSet(detail::parseIntegerSet(inputStr));
}

MultiAffineFunction parseMultiAffineFunction(StringRef str) {
  return detail::getMultiAffineFunctionFromMap(detail::parseAffineMap(str));
}
} // namespace presburger
} // namespace mlir
