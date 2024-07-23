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
#include "clang/Basic/LangOptions.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/ErrorHandling.h"
#include <functional>

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

namespace {
using SVU = llvm::SmallVector<unsigned, 3>;
/// Checks to see if a all the parameters of a template function with a given
/// index refer to the same type.
AST_MATCHER_P(FunctionDecl, areParametersSameTemplateType, SVU, Indexes) {
  const FunctionTemplateDecl *TemplateDecl = Node.getPrimaryTemplate();
  if (!TemplateDecl)
    return false;
  const FunctionDecl *FuncDecl = TemplateDecl->getTemplatedDecl();
  if (!FuncDecl)
    return false;
  assert(!Indexes.empty());
  if (llvm::any_of(Indexes, [Count(FuncDecl->getNumParams())](unsigned Index) {
        return Index >= Count;
      }))
    return false;
  const ParmVarDecl *FirstParam = FuncDecl->getParamDecl(Indexes.front());
  if (!FirstParam)
    return false;
  QualType Type = FirstParam->getOriginalType();
  for (auto Item : llvm::drop_begin(Indexes)) {
    const ParmVarDecl *Param = FuncDecl->getParamDecl(Item);
    if (!Param)
      return false;
    if (Param->getOriginalType() != Type)
      return false;
  }
  return true;
}
AST_MATCHER_P(FunctionDecl, isParameterTypeUnique, unsigned, Index) {
  const FunctionTemplateDecl *TemplateDecl = Node.getPrimaryTemplate();
  if (!TemplateDecl)
    return false;
  const FunctionDecl *FuncDecl = TemplateDecl->getTemplatedDecl();
  if (!FuncDecl)
    return false;
  if (Index >= FuncDecl->getNumParams())
    return false;
  const ParmVarDecl *MainParam = FuncDecl->getParamDecl(Index);
  if (!MainParam)
    return false;
  QualType Type = MainParam->getOriginalType();
  for (unsigned I = 0, E = FuncDecl->getNumParams(); I != E; ++I) {
    if (I == Index)
      continue;
    const ParmVarDecl *Param = FuncDecl->getParamDecl(I);
    if (!Param)
      continue;
    if (Param->getOriginalType() == Type)
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
static constexpr char OutputRangeEnd[] = "OutputRangeEnd";
static constexpr char OutputRangeBegin[] = "OutputRangeBegin";

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

/// Detects a range function where the argument for the begin call differs from
/// the end call
/// @code
///   std::find(I.begin(), J.end());
/// @endcode
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

/// Detects a range function where we expect a call to begin but get a call to
/// end or vice versa
/// @code
///   std::find(X.end(), X.end());
/// @endcode
/// First argument likely should be begin
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

static auto makeHalfOpenMatcher(std::vector<StringRef> FuncNames,
                                unsigned BeginExpected, unsigned PotentialEnd,
                                const FullState &State) {
  auto NameMatcher = hasAnyName(std::move(FuncNames));
  return callExpr(
             anyOf(allOf(callee(functionDecl(
                             NameMatcher, areParametersSameTemplateType(
                                              {BeginExpected, PotentialEnd}))),
                         anyOf(makeRangeArgMismatch(
                                   BeginExpected, PotentialEnd, State.Forward,
                                   State.Reversed, State.MakeReverseIterator),
                               makeUnexpectedBeginEndPair(
                                   BeginExpected, PotentialEnd, State.Combined,
                                   State.MakeReverseIterator))),
                   allOf(callee(functionDecl(NameMatcher, isParameterTypeUnique(
                                                              BeginExpected))),
                         makeUnexpectedBeginEndMatcher(
                             BeginExpected, ArgMismatchBegin,
                             State.Combined.EndNames, State.MakeReverseIterator,
                             State.Combined.BeginNames))))
      .bind(Callee);
}

/// Detects calls where a single output iterator is expected, yet an end of
/// container input is supplied Usually these arguments would be supplied with
/// things like `std::back_inserter`
static auto makeExpectedBeginFullMatcher(std::vector<StringRef> FuncNames,
                                         unsigned ExpectedIndex,
                                         const FullState &State) {
  return callExpr(argumentCountAtLeast(ExpectedIndex + 1),
                  callee(functionDecl(isParameterTypeUnique(ExpectedIndex),
                                      hasAnyName(std::move(FuncNames)))),
                  makeUnexpectedBeginEndMatcher(
                      ExpectedIndex, OutputRangeEnd, State.Combined.EndNames,
                      State.MakeReverseIterator, State.Combined.BeginNames))
      .bind(Callee);
}

/// Detects calls where a single output iterator is expected, yet an end of
/// container input is supplied Usually these arguments would be supplied with
/// things like `std::back_inserter`
static auto makeExpectedEndFullMatcher(std::vector<StringRef> FuncNames,
                                       unsigned ExpectedIndex,
                                       const FullState &State) {
  return callExpr(argumentCountAtLeast(ExpectedIndex + 1),
                  callee(functionDecl(isParameterTypeUnique(ExpectedIndex),
                                      hasAnyName(std::move(FuncNames)))),
                  makeUnexpectedBeginEndMatcher(ExpectedIndex, OutputRangeBegin,
                                                State.Combined.BeginNames,
                                                State.MakeReverseIterator,
                                                State.Combined.EndNames))
      .bind(Callee);
}

/// Detects calls to container methods which expect an argument to be an
/// iterator of the container
/// @code
///   cont_a.erase(cont_b.begin());
/// @endcode
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
                     expr(unless(matchers::isStatementIdenticalToBoundNode(
                              Internal)))
                         .bind(InternalOther),
                     State.All, State.MakeReverseIterator, State.All,
                     ArgMismatchRevBind)
                     .bind(InternalArgument)))
      .bind(Callee);
}

/// Like @c makePairRangeMatcher but instead of operating on free functions,
/// operates on class methods
/// @code
///   cont.insert(cont.begin(), other_range.begin(), other_range.end());
/// @endcode
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

void prependStdPrefix(llvm::MutableArrayRef<std::string> Items) {
  static const StringRef Prefix = "::std::";
  llvm::for_each(Items, [](std::string &item) {
    item.insert(0, Prefix.data(), Prefix.size());
  });
}

/// Gets functions that take 2 whole ranges
static std::vector<std::string>
getSingleRangeWithRevOutputIterator(const LangOptions &LangOpts) {
  std::vector<std::string> Result;
  llvm::append_range(Result, std::array{"copy_backward"});
  if (LangOpts.CPlusPlus11) {
    llvm::append_range(Result, std::array{"move_backward"});
  }
  prependStdPrefix(Result);
  return Result;
}

/// Gets functions that take 2 whole ranges and optionally start with a policy
static std::vector<std::string>
getMultiRangePolicyFunctors(const LangOptions &LangOpts) {
  std::vector<std::string> Result;
  llvm::append_range(Result, std::array{"find_end", "find_first_of", "search",
                                        "includes", "lexicographical_compare"});
  if (LangOpts.CPlusPlus20)
    llvm::append_range(Result, std::array{"lexicographical_compare_three_way"});
  prependStdPrefix(Result);
  return Result;
}

/// Gets a function that takes 2 ranges where the second may be specified by
/// just a start iterator or a start/end pair, The range may optionally start
/// with a policy
static std::vector<std::string>
getMultiRangePolicyPotentiallyHalfOpenFunctors(const LangOptions &LangOpts) {
  std::vector<std::string> Result;
  llvm::append_range(Result, std::array{"mismatch", "equal"});
  prependStdPrefix(Result);
  return Result;
}

static std::vector<std::string>
getMultiRangePotentiallyHalfOpenFunctors(const LangOptions &LangOpts) {
  std::vector<std::string> Result;
  if (LangOpts.CPlusPlus11)
    llvm::append_range(Result, std::array{"is_permutation"});
  prependStdPrefix(Result);
  return Result;
}

static std::vector<std::string>
getMultiRangePolicyWithSingleOutputIterator(const LangOptions &LangOpts) {
  std::vector<std::string> Result;
  llvm::append_range(Result, std::array{"set_union", "set_intersection",
                                        "set_difference",
                                        "set_symmetric_difference", "merge"});
  prependStdPrefix(Result);
  return Result;
}

// Returns a vector of function that take a range in the first and second
// arguments
static std::vector<std::string>
getSingleRangeFunctors(const LangOptions &LangOpts) {
  std::vector<std::string> Result;
  if (LangOpts.CPlusPlus17) {
    llvm::append_range(Result, std::array{"sample"});
  } else {
    llvm::append_range(Result, std::array{"random_shuffle"});
  }
  if (LangOpts.CPlusPlus11) {
    llvm::append_range(Result,
                       std::array{"shuffle", "partition_point", "iota"});
  }
  llvm::append_range(Result, std::array{
                                 "lower_bound",
                                 "upper_bound",
                                 "equal_range",
                                 "binary_search",
                                 "push_heap",
                                 "pop_heap",
                                 "make_heap",
                                 "sort_heap",
                                 "next_permutation",
                                 "prev_permutation",
                                 "accumulate",
                             });
  prependStdPrefix(Result);
  return Result;
}

// Returns a vector of function that take a range in the first and second or
// second and third arguments
static std::vector<std::string>
getSingleRangePolicyFunctors(const LangOptions &LangOpts) {
  std::vector<std::string> Result;
  if (LangOpts.CPlusPlus11)
    llvm::append_range(Result, std::array{
                                   "all_of",
                                   "any_of",
                                   "none_of",
                                   "is_partitioned",
                                   "is_sorted",
                                   "is_sorted_until",
                                   "is_heap",
                                   "is_heap_until",
                                   "minmax_element",
                               });

  if (LangOpts.CPlusPlus17)
    llvm::append_range(Result,
                       std::array{"reduce", "uninitialized_default_construct",
                                  "uninitialized_value_construct", "destroy"});
  if (LangOpts.CPlusPlus20)
    llvm::append_range(Result, std::array{"shift_left", "shift_right"});

  llvm::append_range(Result, std::array{
                                 "find",
                                 "find_if",
                                 "find_if_not",
                                 "adjacent_find",
                                 "count",
                                 "count_if",
                                 "search_n",
                                 "replace",
                                 "replace_if",
                                 "fill",
                                 "generate",
                                 "remove_if",
                                 "unique",
                                 "reverse",
                                 "partition",
                                 "stable_partition",
                                 "sort",
                                 "stable_sort",
                                 "max_element",
                                 "min_element",
                                 "uninitialized_fill",
                             });
  prependStdPrefix(Result);
  return Result;
}

static std::vector<std::string>
getSingleRangePolicyWithSingleOutputIteratorFunctions(
    const LangOptions &LangOpts) {
  std::vector<std::string> Result;
  if (LangOpts.CPlusPlus11)
    llvm::append_range(
        Result,
        std::array{
            "copy", "copy_if", "move", "swap_ranges",
            "partition_copy", // FIXME: This will miss diagnosing the second
                              // range output argument if a policy is specified
        });
  if (LangOpts.CPlusPlus17)
    llvm::append_range(
        Result, std::array{"exclusive_scan", "inclusive_scan",
                           "transform_reduce", "transform_exclusive_scan",
                           "transform_inclusive_scan", "uninitialized_move"});
  llvm::append_range(
      Result,
      std::array{"transform", // FIXME: This is not ideal as transform can take
                              // 2 different input ranges and one output
                 "replace_copy", "replace_copy_if", "remove_copy_if",
                 "unique_copy", "reverse_copy", "adjacent_difference",
                 "uninitialized_copy"

      });
  prependStdPrefix(Result);
  return Result;
}

static std::vector<std::string>
getSingleRangeWithSingleOutputIteratorFunctions(const LangOptions &LangOpts) {
  std::vector<std::string> Result;
  llvm::append_range(
      Result,
      std::array{
          "inner_product", // FIXME: The 3rd argument to this is an input, but
                           // currently the warning says output
          "partial_sum",
      });
  prependStdPrefix(Result);
  return Result;
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
  auto SingleRange = getSingleRangeFunctors(getLangOpts());
  auto SingleRangePolicy = getSingleRangePolicyFunctors(getLangOpts());
  auto SingleRangePolicyHalf =
      getSingleRangePolicyWithSingleOutputIteratorFunctions(getLangOpts());
  auto SingleRangeHalf =
      getSingleRangeWithSingleOutputIteratorFunctions(getLangOpts());
  auto SingleRangeBackwardsHalf =
      getSingleRangeWithRevOutputIterator(getLangOpts());
  auto MultiRangePolicy = getMultiRangePolicyFunctors(getLangOpts());
  auto MultiRangePolicyPotentiallyHalfOpen =
      getMultiRangePolicyPotentiallyHalfOpenFunctors(getLangOpts());
  auto MultiRangePotentiallyHalfOpen =
      getMultiRangePotentiallyHalfOpenFunctors(getLangOpts());

  auto MultiRangePolicySingleOutputIterator =
      getMultiRangePolicyWithSingleOutputIterator(getLangOpts());

  static const auto ToRefs =
      [](std::initializer_list<std::reference_wrapper<std::vector<std::string>>>
             Items) {
        std::vector<StringRef> Result;
        for (const auto &Item : Items) {
          llvm::append_range(Result, Item.get());
        }
        return Result;
      };

  Finder->addMatcher(
      makePairRangeMatcher(
          ToRefs({SingleRange, SingleRangeHalf, SingleRangePolicy,
                  SingleRangePolicyHalf, SingleRangeBackwardsHalf,
                  MultiRangePolicy, MultiRangePotentiallyHalfOpen,
                  MultiRangePolicyPotentiallyHalfOpen,
                  MultiRangePolicySingleOutputIterator}),
          0, 1, State),
      this);
  Finder->addMatcher(
      makePairRangeMatcher(
          ToRefs({SingleRangePolicy, SingleRangePolicyHalf, MultiRangePolicy,
                  MultiRangePolicyPotentiallyHalfOpen,
                  MultiRangePolicySingleOutputIterator}),
          1, 2, State),
      this);
  Finder->addMatcher(
      makeExpectedBeginFullMatcher(
          ToRefs({SingleRangeHalf, SingleRangePolicyHalf}), 2, State),
      this);
  Finder->addMatcher(
      makeExpectedBeginFullMatcher(ToRefs({SingleRangePolicyHalf}), 3, State),
      this);
  Finder->addMatcher(
      makeExpectedEndFullMatcher(ToRefs({SingleRangeBackwardsHalf}), 2, State),
      this);
  Finder->addMatcher(
      makePairRangeMatcher(ToRefs({MultiRangePolicy}), 2, 3, State), this);
  Finder->addMatcher(
      makePairRangeMatcher(ToRefs({MultiRangePolicy}), 3, 4, State), this);
  Finder->addMatcher(
      makeHalfOpenMatcher(ToRefs({MultiRangePotentiallyHalfOpen,
                                  MultiRangePolicyPotentiallyHalfOpen,
                                  MultiRangePolicySingleOutputIterator}),
                          2, 3, State),
      this);
  Finder->addMatcher(
      makeHalfOpenMatcher(ToRefs({MultiRangePolicyPotentiallyHalfOpen,
                                  MultiRangePolicySingleOutputIterator}),
                          3, 4, State),
      this);
  Finder->addMatcher(
      makeExpectedBeginFullMatcher(
          ToRefs({MultiRangePolicySingleOutputIterator}), 4, State),
      this);
  Finder->addMatcher(
      makeExpectedBeginFullMatcher(
          ToRefs({MultiRangePolicySingleOutputIterator}), 5, State),
      this);

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
  } else if (const auto *Range1 = Result.Nodes.getNodeAs<Expr>(FirstRangeArg)) {
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
  } else if (const auto *OutputMismatch =
                 Result.Nodes.getNodeAs<Expr>(OutputRangeEnd)) {
    diag(OutputMismatch->getBeginLoc(),
         "'end' iterator supplied where an output iterator is expected")
        << Result.Nodes.getNodeAs<Expr>(ArgMismatchExpr)->getSourceRange();
    if (const auto *Rev =
            Result.Nodes.getNodeAs<CallExpr>(ArgMismatchRevBind)) {
      diag(Rev->getBeginLoc(), "%0 changes 'begin' into a 'end' iterator",
           DiagnosticIDs::Note)
          << Rev->getSourceRange() << Rev->getDirectCallee();
    }
  } else if (const auto *OutputMismatch =
                 Result.Nodes.getNodeAs<Expr>(OutputRangeBegin)) {
    diag(OutputMismatch->getBeginLoc(),
         "'begin' iterator supplied where an 'end' iterator is expected")
        << Result.Nodes.getNodeAs<Expr>(ArgMismatchExpr)->getSourceRange();
    if (const auto *Rev =
            Result.Nodes.getNodeAs<CallExpr>(ArgMismatchRevBind)) {
      diag(Rev->getBeginLoc(), "%0 changes 'begin' into a 'end' iterator",
           DiagnosticIDs::Note)
          << Rev->getSourceRange() << Rev->getDirectCallee();
    }
  } else {
    llvm_unreachable("Unhandled matches");
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
bool IncorrectIteratorsCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus;
}
} // namespace clang::tidy::bugprone
