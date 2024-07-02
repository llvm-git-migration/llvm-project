//===--- UseRangesCheck.cpp - clang-tidy ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseRangesCheck.h"
#include "Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <limits>
#include <optional>
#include <string>

using namespace clang::ast_matchers;

static constexpr const char BoundCall[] = "CallExpr";
static constexpr const char FuncDecl[] = "FuncDecl";
static constexpr const char ArgName[] = "ArgName";

namespace clang::tidy::utils {

static bool operator==(const UseRangesCheck::Indexes &L,
                       const UseRangesCheck::Indexes &R) {
  return std::tie(L.BeginArg, L.EndArg, L.ReplaceArg) ==
         std::tie(R.BeginArg, R.EndArg, R.ReplaceArg);
}

static std::string getFullPrefix(ArrayRef<UseRangesCheck::Indexes> Signature) {
  std::string Output;
  llvm::raw_string_ostream OS(Output);
  for (auto Item : Signature) {
    OS << Item.BeginArg << ":" << Item.EndArg << ":"
       << (Item.ReplaceArg == Item.First ? '0' : '1');
  }
  return Output;
}

static llvm::hash_code hash_value(const UseRangesCheck::Indexes &Indexes) {
  return llvm::hash_combine(Indexes.BeginArg, Indexes.EndArg,
                            Indexes.ReplaceArg);
}

static llvm::hash_code hash_value(const UseRangesCheck::Signature &Sig) {
  return llvm::hash_combine_range(Sig.begin(), Sig.end());
}

namespace {

AST_MATCHER(Expr, hasSideEffects) {
  return Node.HasSideEffects(Finder->getASTContext());
}
} // namespace

static auto makeMatcher(bool IsBegin, StringRef Prefix) {
  auto Member =
      IsBegin ? expr(unless(hasSideEffects())).bind((ArgName + Prefix).str())
              : expr(matchers::isStatementIdenticalToBoundNode(
                    (ArgName + Prefix).str()));
  return expr(
      anyOf(cxxMemberCallExpr(
                callee(cxxMethodDecl(IsBegin ? hasAnyName("begin", "cbegin")
                                             : hasAnyName("end", "cend"))),
                on(Member)),
            callExpr(argumentCountIs(1), hasArgument(0, Member),
                     hasDeclaration(functionDecl(
                         IsBegin ? hasAnyName("::std::begin", "::std::cbegin")
                                 : hasAnyName("::std::end", "::std::cend"))))));
}
static ast_matchers::internal::Matcher<CallExpr>
makeMatcherPair(StringRef State, const UseRangesCheck::Indexes &Indexes) {
  auto ArgPostfix = std::to_string(Indexes.BeginArg);
  SmallString<64> ID = {BoundCall, State};
  return callExpr(argumentCountAtLeast(
                      std::max(Indexes.BeginArg, Indexes.EndArg) + 1),
                  hasArgument(Indexes.BeginArg, makeMatcher(true, ArgPostfix)),
                  hasArgument(Indexes.EndArg, makeMatcher(false, ArgPostfix)))
      .bind(ID);
}

void UseRangesCheck::registerMatchers(MatchFinder *Finder) {
  Replaces = getReplacerMap();
  llvm::DenseSet<ArrayRef<Signature>> Seen;
  for (auto I = Replaces.begin(), E = Replaces.end(); I != E; ++I) {
    const auto &Replacer = I->getValue();
    const auto &Signatures = Replacer->getReplacementSignatures();
    if (Seen.contains(Signatures))
      continue;
    assert(!Signatures.empty() &&
           llvm::all_of(Signatures, [](auto index) { return !index.empty(); }));
    std::vector<StringRef> Names(1, I->getKey());
    for (auto J = std::next(I); J != E; ++J) {
      if (J->getValue()->getReplacementSignatures() == Signatures) {
        Names.push_back(J->getKey());
      }
    }
    std::vector<ast_matchers::internal::DynTypedMatcher> TotalMatchers;
    // As we match on the first matched signature, we need to sort the
    // signatures in order of length(longest to shortest). This way any
    // signature that is a subset of another signature will be matched after the
    // other.
    SmallVector<Signature> SigVec(Signatures);
    llvm::sort(SigVec, [](auto &L, auto &R) { return R.size() < L.size(); });
    for (const auto &Signature : SigVec) {
      std::vector<ast_matchers::internal::DynTypedMatcher> Matchers;
      for (const auto &ArgPair : Signature) {
        Matchers.push_back(makeMatcherPair(getFullPrefix(Signature), ArgPair));
      }
      TotalMatchers.push_back(
          ast_matchers::internal::DynTypedMatcher::constructVariadic(
              ast_matchers::internal::DynTypedMatcher::VO_AllOf,
              ASTNodeKind::getFromNodeKind<CallExpr>(), std::move(Matchers)));
    }
    Finder->addMatcher(
        callExpr(
            callee(functionDecl(hasAnyName(std::move(Names))).bind(FuncDecl)),
            ast_matchers::internal::DynTypedMatcher::constructVariadic(
                ast_matchers::internal::DynTypedMatcher::VO_AnyOf,
                ASTNodeKind::getFromNodeKind<CallExpr>(),
                std::move(TotalMatchers))
                .convertTo<CallExpr>()),
        this);
  }
}

static void removeFunctionArgs(const CallExpr &Call, ArrayRef<unsigned> Indexes,
                               llvm::SmallVectorImpl<FixItHint> &Output,
                               const ASTContext &Ctx) {
  llvm::SmallVector<unsigned> Sorted(Indexes);
  // Keep track of commas removed
  llvm::SmallBitVector Commas(Call.getNumArgs());
  // The first comma is actually the '(' which we can't remove
  Commas[0] = true;
  llvm::sort(Sorted);
  for (auto Index : Sorted) {
    const auto *Arg = Call.getArg(Index);
    if (Commas[Index]) {
      if (Index >= Commas.size()) {
        Output.push_back(FixItHint::CreateRemoval(Arg->getSourceRange()));
      } else {
        // Remove the next comma
        Commas[Index + 1] = true;
        Output.push_back(
            FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
                {Arg->getBeginLoc(),
                 Lexer::getLocForEndOfToken(Arg->getEndLoc(), 0,
                                            Ctx.getSourceManager(),
                                            Ctx.getLangOpts())
                     .getLocWithOffset(1)})));
      }
    } else {
      Output.push_back(FixItHint::CreateRemoval(CharSourceRange::getTokenRange(
          Arg->getBeginLoc().getLocWithOffset(-1), Arg->getEndLoc())));
      Commas[Index] = true;
    }
  }
}

void UseRangesCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Function = Result.Nodes.getNodeAs<FunctionDecl>(FuncDecl);
  auto QN = "::" + Function->getQualifiedNameAsString();
  auto Iter = Replaces.find(QN);
  assert(Iter != Replaces.end());
  SmallString<64> Buffer;
  for (const auto &Signature : Iter->getValue()->getReplacementSignatures()) {
    Buffer.assign({BoundCall, getFullPrefix(Signature)});
    const auto *Call = Result.Nodes.getNodeAs<CallExpr>(Buffer);
    if (!Call)
      continue;
    auto Diag = createDiag(*Call);
    Diag << FixItHint::CreateReplacement(
        Call->getCallee()->getSourceRange(),
        Iter->getValue()->getReplaceName(*Function));
    if (auto Include = Iter->getValue()->getHeaderInclusion(*Function)) {
      Diag << Inserter.createIncludeInsertion(
          Result.SourceManager->getFileID(Call->getBeginLoc()), *Include);
    }
    llvm::SmallVector<unsigned, 3> ToRemove;
    for (const auto &[First, Second, Replace] : Signature) {
      auto ID = std::to_string(First);
      Diag << FixItHint::CreateReplacement(
          Call->getArg(Replace == Indexes::Second ? Second : First)
              ->getSourceRange(),
          Lexer::getSourceText(
              CharSourceRange::getTokenRange(
                  Result.Nodes.getNodeAs<Expr>(ArgName + ID)->getSourceRange()),
              Result.Context->getSourceManager(),
              Result.Context->getLangOpts()));
      ToRemove.push_back(Replace == Indexes::Second ? First : Second);
    }
    SmallVector<FixItHint> Fixes;
    removeFunctionArgs(*Call, ToRemove, Fixes, *Result.Context);
    Diag << Fixes;
    return;
  }
  llvm_unreachable("No valid signature found");
}

bool UseRangesCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus11;
}

UseRangesCheck::UseRangesCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM),
               areDiagsSelfContained()) {}

void UseRangesCheck::registerPPCallbacks(const SourceManager &,
                                         Preprocessor *PP, Preprocessor *) {
  Inserter.registerPreprocessor(PP);
}

void UseRangesCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
}

std::optional<std::string>
UseRangesCheck::Replacer::getHeaderInclusion(const NamedDecl &) const {
  return std::nullopt;
}

DiagnosticBuilder UseRangesCheck::createDiag(const CallExpr &Call) {
  return diag(Call.getBeginLoc(), "use a ranges version of this algorithm");
}
std::optional<TraversalKind> UseRangesCheck::getCheckTraversalKind() const {
  return TK_IgnoreUnlessSpelledInSource;
}
} // namespace clang::tidy::utils
