//===--- IncorrectIteratorsCheck.cpp - clang-tidy -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IncorrectIteratorsCheck.h"
#include "../utils/Matchers.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTTypeTraits.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {
using SVU = llvm::SmallVector<unsigned, 3>;
/// Checks to see if a function a
AST_MATCHER_P(FunctionDecl, areParametersSameTemplateType, SVU, Indexes) {
  const auto *TemplateDecl = Node.getPrimaryTemplate();
  if (!TemplateDecl)
    return false;
  const auto *FuncDecl = TemplateDecl->getTemplatedDecl();
  if (!FuncDecl)
    return false;
  assert(!Indexes.empty());
  const auto *FirstParam = FuncDecl->getParamDecl(Indexes.front());
  if (!FirstParam)
    return false;
  auto Type = FirstParam->getOriginalType();
  for (auto Item : llvm::drop_begin(Indexes)) {
    const auto *Param = FuncDecl->getParamDecl(Item);
    if (!Param)
      return false;
    if (Param->getOriginalType() != Type)
      return false;
  }
  return true;
}
struct NameMatchers {
  ArrayRef<StringRef> FreeNames;
  ArrayRef<StringRef> MethodNames;
};

struct NamePairs {
  NameMatchers BeginNames;
  NameMatchers EndNames;
};

struct FullState {
  NamePairs Forward;
  NamePairs Reversed;
  NamePairs Combined;
  NameMatchers All;
  ArrayRef<StringRef> MakeReverseIterator;
};

} // namespace

static constexpr char FirstRangeArg[] = "FirstRangeArg";
static constexpr char FirstRangeArgExpr[] = "FirstRangeArgExpr";
static constexpr char ReverseBeginBind[] = "ReverseBeginBind";
static constexpr char ReverseEndBind[] = "ReverseEndBind";
static constexpr char SecondRangeArg[] = "SecondRangeArg";
static constexpr char SecondRangeArgExpr[] = "SecondRangeArgExpr";
static constexpr char ArgMismatchBegin[] = "ArgMismatchBegin";
static constexpr char ArgMismatchEnd[] = "ArgMismatchEnd";
static constexpr char ArgMismatchExpr[] = "ArgMismatchExpr";
static constexpr char ArgMismatchRevBind[] = "ArgMismatchRevBind";
static constexpr char Callee[] = "Callee";
static constexpr char Internal[] = "Internal";
static constexpr char InternalOther[] = "InternalOther";
static constexpr char InternalArgument[] = "InternalArgument";

static auto
makeExprMatcher(ast_matchers::internal::Matcher<Expr> ArgumentMatcher,
                const NameMatchers &Names, ArrayRef<StringRef> MakeReverse,
                const NameMatchers &RevNames, StringRef RevBind) {
  return expr(anyOf(
      cxxMemberCallExpr(argumentCountIs(0),
                        callee(cxxMethodDecl(hasAnyName(Names.MethodNames))),
                        on(ArgumentMatcher)),
      callExpr(argumentCountIs(1),
               hasDeclaration(functionDecl(hasAnyName(Names.FreeNames))),
               hasArgument(0, ArgumentMatcher)),
      callExpr(
          callee(functionDecl(hasAnyName(MakeReverse))), argumentCountIs(1),
          hasArgument(
              0, expr(anyOf(cxxMemberCallExpr(argumentCountIs(0),
                                              callee(cxxMethodDecl(hasAnyName(
                                                  RevNames.MethodNames))),
                                              on(ArgumentMatcher)),
                            callExpr(argumentCountIs(1),
                                     hasDeclaration(functionDecl(
                                         hasAnyName(RevNames.FreeNames))),
                                     hasArgument(0, ArgumentMatcher))))))
          .bind(RevBind)));
}

// Detects a range function where the argument for the begin call differs from
// the end call std::find(I.begin(), J.end())
static auto makeRangeArgMismatch(unsigned BeginExpected, unsigned EndExpected,
                                 NamePairs Forward, NamePairs Reverse,
                                 ArrayRef<StringRef> MakeReverse) {
  std::vector<ast_matchers::internal::DynTypedMatcher> Matchers;

  for (const auto &[Primary, Backwards] :
       {std::pair{Forward, Reverse}, std::pair{Reverse, Forward}}) {
    Matchers.push_back(callExpr(
        hasArgument(BeginExpected,
                    makeExprMatcher(expr().bind(FirstRangeArg),
                                    Primary.BeginNames, MakeReverse,
                                    Backwards.EndNames, ReverseBeginBind)
                        .bind(FirstRangeArgExpr)),
        hasArgument(EndExpected,
                    makeExprMatcher(
                        expr(unless(matchers::isStatementIdenticalToBoundNode(
                                 FirstRangeArg)))
                            .bind(SecondRangeArg),
                        Primary.EndNames, MakeReverse, Backwards.BeginNames,
                        ReverseEndBind)
                        .bind(SecondRangeArgExpr))));
  }

  return callExpr(
      argumentCountAtLeast(std::max(BeginExpected, EndExpected) + 1),
      ast_matchers::internal::DynTypedMatcher::constructVariadic(
          ast_matchers::internal::DynTypedMatcher::VO_AnyOf,
          ASTNodeKind::getFromNodeKind<CallExpr>(), std::move(Matchers))
          .convertTo<CallExpr>());
}

// Detects a range function where we expect a call to begin but get a call to
// end or vice versa std::find(X.end(), X.end()) // Here we would warn on the
// first argument that it should be begin
static auto makeUnexpectedBeginEndMatcher(unsigned Index, StringRef BindTo,
                                          NameMatchers Names,
                                          ArrayRef<StringRef> MakeReverse,
                                          const NameMatchers &RevNames) {
  return callExpr(hasArgument(Index, makeExprMatcher(expr().bind(BindTo), Names,
                                                     MakeReverse, RevNames,
                                                     ArgMismatchRevBind)
                                         .bind(ArgMismatchExpr)));
}

static auto makeUnexpectedBeginEndPair(unsigned BeginExpected,
                                       unsigned EndExpected,
                                       NamePairs BeginEndPairs,
                                       ArrayRef<StringRef> MakeReverse) {
  // return expr(unless(anything()));
  return eachOf(makeUnexpectedBeginEndMatcher(
                    BeginExpected, ArgMismatchBegin, BeginEndPairs.EndNames,
                    MakeReverse, BeginEndPairs.BeginNames),
                makeUnexpectedBeginEndMatcher(
                    EndExpected, ArgMismatchEnd, BeginEndPairs.BeginNames,
                    MakeReverse, BeginEndPairs.EndNames));
}

/// The full matcher for functions that take a range with 2 arguments
static auto makePairRangeMatcher(std::vector<StringRef> FuncNames,
                                 unsigned BeginExpected, unsigned EndExpected,
                                 const FullState &State) {

  return callExpr(callee(functionDecl(hasAnyName(std::move(FuncNames)),
                                      areParametersSameTemplateType(
                                          {BeginExpected, EndExpected}))),
                  anyOf(makeRangeArgMismatch(BeginExpected, EndExpected,
                                             State.Forward, State.Reversed,
                                             State.MakeReverseIterator),
                        makeUnexpectedBeginEndPair(BeginExpected, EndExpected,
                                                   State.Combined,
                                                   State.MakeReverseIterator)))
      .bind(Callee);
}

static auto makeContainerInternalMatcher(std::vector<StringRef> ClassNames,
                                         std::vector<StringRef> ClassMethods,
                                         unsigned InternalExpected,
                                         const FullState &State) {
  return cxxMemberCallExpr(
      thisPointerType(cxxRecordDecl(hasAnyName(std::move(ClassNames)))),
      callee(cxxMethodDecl(hasAnyName(std::move(ClassMethods)))),
      on(expr().bind(Internal)),
      hasArgument(
          InternalExpected,
          makeExprMatcher(
              expr(unless(matchers::isStatementIdenticalToBoundNode(Internal)))
                  .bind(InternalOther),
              State.All, State.MakeReverseIterator, State.All,
              ArgMismatchRevBind)
              .bind(InternalArgument))).bind(Callee);
}

/// Full matcher for class methods that take a range with 2 arguments
static auto makeContainerPairRangeMatcher(std::vector<StringRef> ClassNames,
                                          std::vector<StringRef> ClassMethods,
                                          unsigned BeginExpected,
                                          unsigned EndExpected,
                                          const FullState &State) {
  return cxxMemberCallExpr(
             thisPointerType(cxxRecordDecl(hasAnyName(std::move(ClassNames)))),
             callee(cxxMethodDecl(
                 hasAnyName(std::move(ClassMethods)),
                 areParametersSameTemplateType({BeginExpected, EndExpected}))),
             anyOf(makeRangeArgMismatch(BeginExpected, EndExpected,
                                        State.Forward, State.Reversed,
                                        State.MakeReverseIterator),
                   makeUnexpectedBeginEndPair(BeginExpected, EndExpected,
                                              State.Combined,
                                              State.MakeReverseIterator)))
      .bind(Callee);
}

void IncorrectIteratorsCheck::registerMatchers(MatchFinder *Finder) {
  NamePairs Forward{NameMatchers{BeginFree, BeginMethod},
                    NameMatchers{EndFree, EndMethod}};
  NamePairs Reverse{NameMatchers{RBeginFree, RBeginMethod},
                    NameMatchers{REndFree, REndMethod}};
  llvm::SmallVector<StringRef, 8> CombinedFreeBegin{
      llvm::iterator_range{llvm::concat<StringRef>(BeginFree, RBeginFree)}};
  llvm::SmallVector<StringRef, 8> CombinedFreeEnd{
      llvm::iterator_range{llvm::concat<StringRef>(EndFree, REndFree)}};
  llvm::SmallVector<StringRef, 8> CombinedMethodBegin{
      llvm::iterator_range{llvm::concat<StringRef>(BeginMethod, RBeginMethod)}};
  llvm::SmallVector<StringRef, 8> CombinedMethodEnd{
      llvm::iterator_range{llvm::concat<StringRef>(EndMethod, REndMethod)}};
  llvm::SmallVector<StringRef, 16> AllFree{llvm::iterator_range{
      llvm::concat<StringRef>(CombinedFreeBegin, CombinedFreeEnd)}};
  llvm::SmallVector<StringRef, 16> AllMethod{llvm::iterator_range{
      llvm::concat<StringRef>(CombinedMethodBegin, CombinedMethodEnd)}};
  NamePairs Combined{NameMatchers{CombinedFreeBegin, CombinedMethodBegin},
                     NameMatchers{CombinedFreeEnd, CombinedMethodEnd}};
  FullState State{
      Forward, Reverse, Combined, {AllFree, AllMethod}, MakeReverseIterator};
  Finder->addMatcher(makePairRangeMatcher({"::std::find"}, 1, 2, State), this);
  Finder->addMatcher(makePairRangeMatcher({"::std::find"}, 0, 1, State), this);
  Finder->addMatcher(
      makeContainerPairRangeMatcher({"::std::vector", "::std::list"},
                                    {"insert"}, 1, 2, State),
      this);
  Finder->addMatcher(
      makeContainerInternalMatcher({"::std::vector", "::std::list"},
                                   {"insert", "erase", "emplace"}, 0, State),
      this);
}

void IncorrectIteratorsCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>(Callee);
  if (const auto *BeginMismatch =
          Result.Nodes.getNodeAs<Expr>(ArgMismatchBegin)) {
    diag(BeginMismatch->getBeginLoc(),
         "'end' iterator supplied where a 'begin' iterator is expected")
        << Result.Nodes.getNodeAs<Expr>(ArgMismatchExpr)->getSourceRange();
    if (const auto *Rev =
            Result.Nodes.getNodeAs<CallExpr>(ArgMismatchRevBind)) {
      diag(Rev->getBeginLoc(), "%0 changes 'begin' into a 'end' iterator",
           DiagnosticIDs::Note)
          << Rev->getSourceRange() << Rev->getDirectCallee();
    }
  } else if (const auto *EndMismatch =
                 Result.Nodes.getNodeAs<Expr>(ArgMismatchEnd)) {
    diag(EndMismatch->getBeginLoc(),
         "'begin' iterator supplied where an 'end' iterator is expected")
        << Result.Nodes.getNodeAs<Expr>(ArgMismatchExpr)->getSourceRange();
    if (const auto *Rev =
            Result.Nodes.getNodeAs<CallExpr>(ArgMismatchRevBind)) {
      diag(Rev->getBeginLoc(), "%0 changes 'end' into a 'begin' iterator",
           DiagnosticIDs::Note)
          << Rev->getSourceRange() << Rev->getDirectCallee();
    }
  } else if (const auto *InternalArg =
                 Result.Nodes.getNodeAs<Expr>(InternalArgument)) {
    diag(InternalArg->getBeginLoc(),
         "%0 called with an iterator for a different container")
        << Call->getDirectCallee();
    const auto *Object = Result.Nodes.getNodeAs<Expr>(Internal);
    diag(Object->getBeginLoc(), "container is specified here",
         DiagnosticIDs::Note)
        << Object->getSourceRange();
    const auto *Other = Result.Nodes.getNodeAs<Expr>(InternalOther);
    diag(Other->getBeginLoc(), "different container provided here",
         DiagnosticIDs::Note)
        << Other->getSourceRange();
  } else {
    const auto *Range1 = Result.Nodes.getNodeAs<Expr>(FirstRangeArg);
    const auto *Range2 = Result.Nodes.getNodeAs<Expr>(SecondRangeArg);
    const auto *FullRange1 = Result.Nodes.getNodeAs<Expr>(FirstRangeArgExpr);
    const auto *FullRange2 = Result.Nodes.getNodeAs<Expr>(SecondRangeArgExpr);
    assert(Range1 && Range2 && FullRange1 && FullRange2 && "Unexpected match");
    diag(Call->getBeginLoc(), "mismatched ranges pass to function")
        << FullRange1->getSourceRange() << FullRange2->getSourceRange();
    diag(Range1->getBeginLoc(), "first range passed here to begin",
         DiagnosticIDs::Note)
        << FullRange1->getSourceRange();
    diag(Range2->getBeginLoc(), "different range passed here to end",
         DiagnosticIDs::Note)
        << FullRange2->getSourceRange();
  }
}

IncorrectIteratorsCheck::IncorrectIteratorsCheck(StringRef Name,
                                                 ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      BeginFree(utils::options::parseStringList(
          Options.get("BeginFree", "::std::begin;::std::cbegin"))),
      EndFree(utils::options::parseStringList(
          Options.get("EndFree", "::std::end;::std::cend"))),
      BeginMethod(utils::options::parseStringList(
          Options.get("BeginMethod", "begin;cbegin"))),
      EndMethod(utils::options::parseStringList(
          Options.get("EndMethod", "end;cend"))),
      RBeginFree(utils::options::parseStringList(
          Options.get("RBeginFree", "::std::rbegin;::std::crbegin"))),
      REndFree(utils::options::parseStringList(
          Options.get("REndFree", "::std::rend;::std::crend"))),
      RBeginMethod(utils::options::parseStringList(
          Options.get("RBeginMethod", "rbegin;crbegin"))),
      REndMethod(utils::options::parseStringList(
          Options.get("REndMethod", "rend;crend"))),
      MakeReverseIterator(utils::options::parseStringList(
          Options.get("MakeReverseIterator", "::std::make_reverse_iterator"))) {
}

void IncorrectIteratorsCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "BeginFree",
                utils::options::serializeStringList(BeginFree));
  Options.store(Opts, "EndFree", utils::options::serializeStringList(EndFree));
  Options.store(Opts, "BeginMethod",
                utils::options::serializeStringList(BeginMethod));
  Options.store(Opts, "EndMethod",
                utils::options::serializeStringList(EndMethod));
  Options.store(Opts, "RBeginFree",
                utils::options::serializeStringList(RBeginFree));
  Options.store(Opts, "REndFree",
                utils::options::serializeStringList(REndFree));
  Options.store(Opts, "RBeginMethod",
                utils::options::serializeStringList(RBeginMethod));
  Options.store(Opts, "REndMethod",
                utils::options::serializeStringList(REndMethod));
  Options.store(Opts, "MakeReverseIterator",
                utils::options::serializeStringList(MakeReverseIterator));
}
std::optional<TraversalKind>
IncorrectIteratorsCheck::getCheckTraversalKind() const {
  return TK_IgnoreUnlessSpelledInSource;
}
} // namespace clang::tidy::bugprone
