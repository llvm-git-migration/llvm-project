
#include "clang/Frontend/CodeSnippetHighlighter.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

static SourceManager createTempSourceManager() {
  FileSystemOptions FileOpts;
  FileManager FileMgr(FileOpts);
  llvm::IntrusiveRefCntPtr<DiagnosticIDs> DiagIDs(new DiagnosticIDs());
  llvm::IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts(new DiagnosticOptions());
  DiagnosticsEngine diags(DiagIDs, DiagOpts);
  return SourceManager(diags, FileMgr);
}

static Lexer createTempLexer(llvm::MemoryBufferRef B, SourceManager &FakeSM,
                             const LangOptions &LangOpts) {
  return Lexer(FakeSM.createFileID(B), B, FakeSM, LangOpts);
}

std::vector<StyleRange> CodeSnippetHighlighter::highlightLine(
    StringRef SourceLine, const Preprocessor *PP, const LangOptions &LangOpts) {
  constexpr raw_ostream::Colors CommentColor = raw_ostream::BLACK;
  constexpr raw_ostream::Colors LiteralColor = raw_ostream::GREEN;
  constexpr raw_ostream::Colors KeywordColor = raw_ostream::YELLOW;

  SourceManager FakeSM = createTempSourceManager();
  const auto MemBuf = llvm::MemoryBuffer::getMemBuffer(SourceLine);
  Lexer L = createTempLexer(MemBuf->getMemBufferRef(), FakeSM, LangOpts);
  L.SetKeepWhitespaceMode(true);

  std::vector<StyleRange> Styles;
  bool Stop = false;
  while (!Stop) {
    Token T;
    Stop = L.LexFromRawLexer(T);
    if (T.is(tok::unknown))
      continue;

    bool Invalid;
    unsigned Start =
        FakeSM.getSpellingColumnNumber(T.getLocation(), &Invalid) - 1;
    if (Invalid)
      continue;

    if (T.is(tok::raw_identifier)) {
      StringRef RawIdent = T.getRawIdentifier();
      // Special case true/false/nullptr literals, since they will otherwise be
      // treated as keywords.
      if (RawIdent == "true" || RawIdent == "false" || RawIdent == "nullptr") {
        Styles.push_back(
            StyleRange{Start, Start + T.getLength(), LiteralColor});
      } else {
        const IdentifierInfo *II = PP->getIdentifierInfo(RawIdent);
        assert(II);

        if (II->isKeyword(LangOpts)) {
          Styles.push_back(
              StyleRange{Start, Start + T.getLength(), KeywordColor});
        }
      }
    } else if (tok::isLiteral(T.getKind())) {
      Styles.push_back(StyleRange{Start, Start + T.getLength(), LiteralColor});
    } else if (T.is(tok::comment)) {
      Styles.push_back(StyleRange{Start, Start + T.getLength(), CommentColor});
    }
  }

  return Styles;
}
