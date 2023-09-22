
#include "clang/Frontend/CodeSnippetHighlighter.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

static constexpr raw_ostream::Colors CommentColor = raw_ostream::GREEN;
static constexpr raw_ostream::Colors LiteralColor = raw_ostream::CYAN;
static constexpr raw_ostream::Colors KeywordColor = raw_ostream::BLUE;

std::vector<StyleRange> CodeSnippetHighlighter::highlightLine(
    unsigned LineNumber, const Preprocessor *PP, const LangOptions &LangOpts,
    FileID FID, const SourceManager &SM) {
  if (!PP)
    return {};

  // Might cause emission of another diagnostic.
  if (PP->getIdentifierTable().getExternalIdentifierLookup())
    return {};

  // Classify the given token and append it to the given vector.
  auto appendStyle = [PP, &LangOpts](std::vector<StyleRange> &Vec,
                                     const Token &T, unsigned Start,
                                     unsigned Length) -> void {
    if (T.is(tok::raw_identifier)) {
      StringRef RawIdent = T.getRawIdentifier();
      // Special case true/false/nullptr literals, since they will otherwise be
      // treated as keywords.
      if (RawIdent == "true" || RawIdent == "false" || RawIdent == "nullptr") {
        Vec.push_back(StyleRange{Start, Start + Length, LiteralColor});
      } else {
        const IdentifierInfo *II = PP->getIdentifierInfo(RawIdent);
        assert(II);

        if (II->isKeyword(LangOpts)) {
          Vec.push_back(StyleRange{Start, Start + Length, KeywordColor});
        }
      }
    } else if (tok::isLiteral(T.getKind())) {
      Vec.push_back(StyleRange{Start, Start + Length, LiteralColor});
    } else if (T.is(tok::comment)) {
      Vec.push_back(StyleRange{Start, Start + Length, CommentColor});
    }
  };

  auto Buff = SM.getBufferOrNone(FID);
  assert(Buff);
  Lexer L = Lexer(FID, *Buff, SM, LangOpts);
  L.SetKeepWhitespaceMode(true);
  std::vector<std::vector<StyleRange>> Lines;

  bool Stop = false;
  while (!Stop) {
    Token T;
    Stop = L.LexFromRawLexer(T);
    if (T.is(tok::unknown))
      continue;

    bool Invalid;
    unsigned StartCol =
        SM.getSpellingColumnNumber(T.getLocation(), &Invalid) - 1;
    if (Invalid)
      continue;
    unsigned StartLine =
        SM.getSpellingLineNumber(T.getLocation(), &Invalid) - 1;
    if (Invalid)
      continue;

    while (Lines.size() <= StartLine)
      Lines.push_back({});

    unsigned EndLine = SM.getSpellingLineNumber(T.getEndLoc(), &Invalid) - 1;
    if (Invalid)
      continue;

    // Simple tokens.
    if (StartLine == EndLine) {
      appendStyle(Lines[StartLine], T, StartCol, T.getLength());
      continue;
    }
    unsigned NumLines = EndLine - StartLine;

    // For tokens that span multiple lines (think multiline comments), we
    // divide them into multiple StyleRanges.
    unsigned EndCol = SM.getSpellingColumnNumber(T.getEndLoc(), &Invalid) - 1;
    if (Invalid)
      continue;

    std::string Spelling = Lexer::getSpelling(T, SM, LangOpts);

    unsigned L = 0;
    unsigned LineLength = 0;
    for (unsigned I = 0; I <= Spelling.size(); ++I) {
      // This line is done.
      if (Spelling[I] == '\n' || Spelling[I] == '\r' || I == Spelling.size()) {
        while (Lines.size() <= StartLine + L)
          Lines.push_back({});

        if (L == 0) // First line
          appendStyle(Lines[StartLine + L], T, StartCol, LineLength);
        else if (L == NumLines) // Last line
          appendStyle(Lines[StartLine + L], T, 0, EndCol);
        else
          appendStyle(Lines[StartLine + L], T, 0, LineLength);
        ++L;
        LineLength = 0;
        continue;
      }
      ++LineLength;
    }
  }

#if 0
  llvm::errs() << "--\nLine Style info: \n";
  int I = 0;
  for (std::vector<StyleRange> &Line : Lines) {
    llvm::errs() << I << ": ";
    for (const auto &R : Line) {
      llvm::errs() << "{" << R.Start << ", " << R.End << "}, ";
    }
    llvm::errs() << "\n";

    ++I;
  }
#endif

  while (Lines.size() <= LineNumber)
    Lines.push_back({});
  return Lines[LineNumber];
}
