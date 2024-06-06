//===- Lexer.cpp - Presburger Lexer Implementation ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the lexer for the Presburger textual form.
//
//===----------------------------------------------------------------------===//

#include "Lexer.h"
#include "Token.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir::presburger;

Lexer::Lexer(const llvm::SourceMgr &sourceMgr) : sourceMgr(sourceMgr) {
  auto bufferID = sourceMgr.getMainFileID();
  curBuffer = sourceMgr.getMemoryBuffer(bufferID)->getBuffer();
  curPtr = curBuffer.begin();
}

/// emitError - Emit an error message and return an Token::error token.
Token Lexer::emitError(const char *loc, const llvm::Twine &message) {
  sourceMgr.PrintMessage(SMLoc::getFromPointer(loc), llvm::SourceMgr::DK_Error,
                         message);
  return formToken(Token::error, loc);
}

Token Lexer::lexToken() {
  while (true) {
    const char *tokStart = curPtr;

    // Lex the next token.
    switch (*curPtr++) {
    default:
      // Handle bare identifiers.
      if (isalpha(curPtr[-1]))
        return lexBareIdentifierOrKeyword(tokStart);

      // Unknown character, emit an error.
      return emitError(tokStart, "unexpected character");

    case ' ':
    case '\t':
    case '\n':
    case '\r':
      // Handle whitespace.
      continue;

    case '_':
      // Handle bare identifiers.
      return lexBareIdentifierOrKeyword(tokStart);

    case 0:
      // This may either be a nul character in the source file or may be the EOF
      // marker that llvm::MemoryBuffer guarantees will be there.
      if (curPtr - 1 == curBuffer.end())
        return formToken(Token::eof, tokStart);
      continue;

    case ':':
      return formToken(Token::colon, tokStart);
    case ',':
      return formToken(Token::comma, tokStart);
    case '(':
      return formToken(Token::l_paren, tokStart);
    case ')':
      return formToken(Token::r_paren, tokStart);
    case '{':
      return formToken(Token::l_brace, tokStart);
    case '}':
      return formToken(Token::r_brace, tokStart);
    case '[':
      return formToken(Token::l_square, tokStart);
    case ']':
      return formToken(Token::r_square, tokStart);
    case '<':
      return formToken(Token::less, tokStart);
    case '>':
      return formToken(Token::greater, tokStart);
    case '=':
      return formToken(Token::equal, tokStart);
    case '+':
      return formToken(Token::plus, tokStart);
    case '*':
      return formToken(Token::star, tokStart);
    case '-':
      if (*curPtr == '>') {
        ++curPtr;
        return formToken(Token::arrow, tokStart);
      }
      return formToken(Token::minus, tokStart);
    case '0':
    case '1':
    case '2':
    case '3':
    case '4':
    case '5':
    case '6':
    case '7':
    case '8':
    case '9':
      return lexNumber(tokStart);
    }
  }
}

/// Lex a bare identifier or keyword that starts with a letter.
///
///   bare-id ::= (letter|[_]) (letter|digit|[_$.])*
///   integer-type ::= `[su]?i[1-9][0-9]*`
///
Token Lexer::lexBareIdentifierOrKeyword(const char *tokStart) {
  // Match the rest of the identifier regex: [0-9a-zA-Z_.$]*
  while (isalpha(*curPtr) || isdigit(*curPtr) || *curPtr == '_' ||
         *curPtr == '$' || *curPtr == '.')
    ++curPtr;

  // Check to see if this identifier is a keyword.
  StringRef spelling(tokStart, curPtr - tokStart);

  auto isAllDigit = [](StringRef str) {
    return llvm::all_of(str, llvm::isDigit);
  };

  // Check for i123, si456, ui789.
  if ((spelling.size() > 1 && tokStart[0] == 'i' &&
       isAllDigit(spelling.drop_front())) ||
      ((spelling.size() > 2 && tokStart[1] == 'i' &&
        (tokStart[0] == 's' || tokStart[0] == 'u')) &&
       isAllDigit(spelling.drop_front(2))))
    return Token(Token::inttype, spelling);

  Token::Kind kind = llvm::StringSwitch<Token::Kind>(spelling)
#define TOK_KEYWORD(SPELLING) .Case(#SPELLING, Token::kw_##SPELLING)
#include "TokenKinds.def"
                         .Default(Token::bare_identifier);

  return Token(kind, spelling);
}

/// Lex a number literal.
///
///   integer-literal ::= digit+ | `0x` hex_digit+
///   float-literal ::= [-+]?[0-9]+[.][0-9]*([eE][-+]?[0-9]+)?
///
Token Lexer::lexNumber(const char *tokStart) {
  assert(isdigit(curPtr[-1]));

  // Handle the normal decimal case.
  while (isdigit(*curPtr))
    ++curPtr;

  return formToken(Token::integer, tokStart);
}
