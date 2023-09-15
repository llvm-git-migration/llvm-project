
#include "clang/Frontend/CodeSnippetHighlighter.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

void CodeSnippetHighlighter::ensureTokenData() {
  if (Initialized)
    return;

  // List of keywords, literals and types we want to highlight.
  // These are best-effort, as is everything we do wrt. highlighting.
  Keywords.insert("_Static_assert");
  Keywords.insert("auto");
  Keywords.insert("concept");
  Keywords.insert("const");
  Keywords.insert("consteval");
  Keywords.insert("constexpr");
  Keywords.insert("delete");
  Keywords.insert("do");
  Keywords.insert("else");
  Keywords.insert("final");
  Keywords.insert("for");
  Keywords.insert("if");
  Keywords.insert("mutable");
  Keywords.insert("namespace");
  Keywords.insert("new");
  Keywords.insert("private");
  Keywords.insert("public");
  Keywords.insert("requires");
  Keywords.insert("return");
  Keywords.insert("static");
  Keywords.insert("static_assert");
  Keywords.insert("using");
  Keywords.insert("void");
  Keywords.insert("volatile");
  Keywords.insert("while");

  // Builtin types we highlight
  Keywords.insert("void");
  Keywords.insert("char");
  Keywords.insert("short");
  Keywords.insert("int");
  Keywords.insert("unsigned");
  Keywords.insert("long");
  Keywords.insert("float");
  Keywords.insert("double");

  Literals.insert("true");
  Literals.insert("false");
  Literals.insert("nullptr");

  Initialized = true;
}

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

std::vector<StyleRange>
CodeSnippetHighlighter::highlightLine(StringRef SourceLine,
                                      const LangOptions &LangOpts) {
  ensureTokenData();

  constexpr raw_ostream::Colors CommentColor = raw_ostream::BLACK;
  constexpr raw_ostream::Colors LiteralColor = raw_ostream::GREEN;
  constexpr raw_ostream::Colors KeywordColor = raw_ostream::YELLOW;

  const auto MemBuf = llvm::MemoryBuffer::getMemBuffer(SourceLine);
  SourceManager FakeSM = createTempSourceManager();
  Lexer L = createTempLexer(MemBuf->getMemBufferRef(), FakeSM, LangOpts);
  L.SetKeepWhitespaceMode(true);

  std::vector<StyleRange> Styles;
  bool Stop = false;
  while (!Stop) {
    Token tok;
    Stop = L.LexFromRawLexer(tok);
    if (tok.is(tok::unknown))
      continue;

    bool Invalid;
    unsigned Start =
        FakeSM.getSpellingColumnNumber(tok.getLocation(), &Invalid) - 1;
    if (Invalid)
      continue;

    if (tok.is(tok::raw_identifier)) {
      // Almost everything we lex is an identifier, since we use a raw lexer.
      // Some should be highlightes as literals, others as keywords.
      if (Keywords.contains(tok.getRawIdentifier()))
        Styles.push_back(
            StyleRange{Start, Start + tok.getLength(), KeywordColor});
      else if (Literals.contains(tok.getRawIdentifier()))
        Styles.push_back(
            StyleRange{Start, Start + tok.getLength(), LiteralColor});
    } else if (tok::isLiteral(tok.getKind())) {
      Styles.push_back(
          StyleRange{Start, Start + tok.getLength(), LiteralColor});
    } else if (tok.is(tok::comment)) {
      Styles.push_back(
          StyleRange{Start, Start + tok.getLength(), CommentColor});
    }
  }

  return Styles;
}
