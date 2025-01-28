#include "clang/Parse/ParseHLSLRootSignature.h"

using namespace llvm::hlsl::root_signature;

namespace clang {
namespace hlsl {

// Lexer Definitions

static bool IsNumberChar(char C) {
  // TODO(#120472): extend for float support exponents
  return isdigit(C); // integer support
}

bool RootSignatureLexer::LexNumber(RootSignatureToken &Result) {
  // NumericLiteralParser does not handle the sign so we will manually apply it
  bool Negative = Buffer.front() == '-';
  bool Signed = Negative || Buffer.front() == '+';
  if (Signed)
    AdvanceBuffer();

  // Retrieve the possible number
  StringRef NumSpelling = Buffer.take_while(IsNumberChar);

  // Catch this now as the Literal Parser will accept it as valid
  if (NumSpelling.empty()) {
    PP.getDiagnostics().Report(Result.TokLoc,
                               diag::err_hlsl_invalid_number_literal);
    return true;
  }

  // Parse the numeric value and do semantic checks on its specification
  clang::NumericLiteralParser Literal(NumSpelling, SourceLoc,
                                      PP.getSourceManager(), PP.getLangOpts(),
                                      PP.getTargetInfo(), PP.getDiagnostics());
  if (Literal.hadError)
    return true; // Error has already been reported so just return

  if (!Literal.isIntegerLiteral()) {
    // Note: if IsNumberChar allows for hexidecimal we will need to turn this
    // into a diagnostics for potential fixed-point literals
    llvm_unreachable("IsNumberChar will only support digits");
    return true;
  }

  // Retrieve the number value to store into the token
  Result.Kind = TokenKind::int_literal;

  llvm::APSInt X = llvm::APSInt(32, !Signed);
  if (Literal.GetIntegerValue(X)) {
    // Report that the value has overflowed
    PP.getDiagnostics().Report(Result.TokLoc,
                               diag::err_hlsl_number_literal_overflow)
        << (unsigned)Signed << NumSpelling;
    return true;
  }

  X = Negative ? -X : X;
  Result.NumLiteral = APValue(X);

  AdvanceBuffer(NumSpelling.size());
  return false;
}

bool RootSignatureLexer::Lex(SmallVector<RootSignatureToken> &Tokens) {
  // Discard any leading whitespace
  AdvanceBuffer(Buffer.take_while(isspace).size());

  while (!Buffer.empty()) {
    // Record where this token is in the text for usage in parser diagnostics
    RootSignatureToken Result(SourceLoc);
    if (LexToken(Result))
      return true;

    // Successfully Lexed the token so we can store it
    Tokens.push_back(Result);

    // Discard any trailing whitespace
    AdvanceBuffer(Buffer.take_while(isspace).size());
  }

  return false;
}

bool RootSignatureLexer::LexToken(RootSignatureToken &Result) {
  char C = Buffer.front();

  // Punctuators
  switch (C) {
#define PUNCTUATOR(X, Y)                                                       \
  case Y: {                                                                    \
    Result.Kind = TokenKind::pu_##X;                                           \
    AdvanceBuffer();                                                           \
    return false;                                                              \
  }
#include "clang/Parse/HLSLRootSignatureTokenKinds.def"
  default:
    break;
  }

  // Numeric constant
  if (isdigit(C) || C == '-' || C == '+')
    return LexNumber(Result);

  // All following tokens require at least one additional character
  if (Buffer.size() <= 1) {
    PP.getDiagnostics().Report(Result.TokLoc, diag::err_hlsl_invalid_token);
    return true;
  }

  // Peek at the next character to deteremine token type
  char NextC = Buffer[1];

  // Registers: [tsub][0-9+]
  if ((C == 't' || C == 's' || C == 'u' || C == 'b') && isdigit(NextC)) {
    AdvanceBuffer();

    if (LexNumber(Result))
      return true; // Error parsing number which is already reported

    // Lex number could also parse a float so ensure it was an unsigned int
    if (Result.Kind != TokenKind::int_literal ||
        Result.NumLiteral.getInt().isSigned()) {
      // Return invalid number literal for register error
      PP.getDiagnostics().Report(Result.TokLoc,
                                 diag::err_hlsl_invalid_register_literal);
      return true;
    }

    // Convert character to the register type.
    // This is done after LexNumber to override the TokenKind
    switch (C) {
    case 'b':
      Result.Kind = TokenKind::bReg;
      break;
    case 't':
      Result.Kind = TokenKind::tReg;
      break;
    case 'u':
      Result.Kind = TokenKind::uReg;
      break;
    case 's':
      Result.Kind = TokenKind::sReg;
      break;
    default:
      llvm_unreachable("Switch for an expected token was not provided");
      return true;
    }
    return false;
  }

  // Keywords and Enums:
  StringRef TokSpelling =
      Buffer.take_while([](char C) { return isalnum(C) || C == '_'; });

  // Define a large string switch statement for all the keywords and enums
  auto Switch = llvm::StringSwitch<TokenKind>(TokSpelling);
#define KEYWORD(NAME) Switch.Case(#NAME, TokenKind::kw_##NAME);
#define ENUM(NAME, LIT) Switch.CaseLower(LIT, TokenKind::en_##NAME);
#include "clang/Parse/HLSLRootSignatureTokenKinds.def"

  // Then attempt to retreive a string from it
  auto Kind = Switch.Default(TokenKind::invalid);
  if (Kind == TokenKind::invalid) {
    PP.getDiagnostics().Report(Result.TokLoc, diag::err_hlsl_invalid_token);
    return true;
  }

  Result.Kind = Kind;
  AdvanceBuffer(TokSpelling.size());
  return false;
}

// Parser Definitions

RootSignatureParser::RootSignatureParser(
    SmallVector<RootElement> &Elements,
    const SmallVector<RootSignatureToken> &Tokens, DiagnosticsEngine &Diags)
    : Elements(Elements), Diags(Diags) {
  CurTok = Tokens.begin();
  LastTok = Tokens.end();
}

bool RootSignatureParser::Parse() {
  // Handle edge-case of empty RootSignature()
  if (CurTok == LastTok)
    return false;

  bool First = true;
  // Iterate as many RootElements as possible
  while (!ParseRootElement(First)) {
    First = false;
    // Avoid use of ConsumeNextToken here to skip incorrect end of tokens error
    CurTok++;
    if (CurTok == LastTok)
      return false;
    if (EnsureExpectedToken(TokenKind::pu_comma))
      return true;
  }

  return true;
}

bool RootSignatureParser::ParseRootElement(bool First) {
  if (First && EnsureExpectedToken(TokenKind::kw_DescriptorTable))
    return true;
  if (!First && ConsumeExpectedToken(TokenKind::kw_DescriptorTable))
    return true;

  // Dispatch onto the correct parse method
  switch (CurTok->Kind) {
  case TokenKind::kw_DescriptorTable:
    return ParseDescriptorTable();
  default:
    llvm_unreachable("Switch for an expected token was not provided");
    return true;
  }
}

bool RootSignatureParser::ParseDescriptorTable() {
  DescriptorTable Table;

  if (ConsumeExpectedToken(TokenKind::pu_l_paren))
    return true;

  // Empty case:
  if (!ConsumeExpectedToken(TokenKind::pu_r_paren)) {
    Elements.push_back(Table);
    return false;
  }

  return true;

}

bool RootSignatureParser::ConsumeNextToken() {
  CurTok++;
  if (LastTok == CurTok) {
    return true;
  }
  return false;
}

// Is given token one of the expected kinds
static bool IsExpectedToken(TokenKind Kind, ArrayRef<TokenKind> AnyExpected) {
  for (auto Expected : AnyExpected)
    if (Kind == Expected)
      return true;
  return false;
}

bool RootSignatureParser::EnsureExpectedToken(TokenKind Expected) {
  return EnsureExpectedToken(ArrayRef{Expected});
}

bool RootSignatureParser::EnsureExpectedToken(ArrayRef<TokenKind> AnyExpected) {
  if (IsExpectedToken(CurTok->Kind, AnyExpected))
    return false;

  return true;
}

bool RootSignatureParser::ConsumeExpectedToken(TokenKind Expected) {
  return ConsumeExpectedToken(ArrayRef{Expected});
}

bool RootSignatureParser::ConsumeExpectedToken(
    ArrayRef<TokenKind> AnyExpected) {
  if (ConsumeNextToken())
    return true;

  return EnsureExpectedToken(AnyExpected);
}

} // namespace hlsl
} // namespace clang
