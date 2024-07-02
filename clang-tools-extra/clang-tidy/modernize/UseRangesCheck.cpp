//===--- UseRangesCheck.cpp - clang-tidy ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseRangesCheck.h"
#include "clang/AST/Decl.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <initializer_list>

// FixItHint - Let the docs script know that this class does provide fixits

namespace clang::tidy::modernize {

static constexpr const char *SingleRangeNames[] = {
    "all_of",
    "any_of",
    "none_of",
    "for_each",
    "find",
    "find_if",
    "find_if_not",
    "adjacent_find",
    "copy",
    "copy_if",
    "copy_backward",
    "move",
    "move_backward",
    "fill",
    "transform",
    "replace",
    "replace_if",
    "generate",
    "remove",
    "remove_if",
    "remove_copy",
    "remove_copy_if",
    "unique",
    "unique_copy",
    "sample",
    "partition_point",
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
    "iota",
};

static constexpr const char *SingleRangeWithExecNames[] = {
    "reverse",
    "reverse_copy",
    "shift_left",
    "shift_right",
    "is_partitioned",
    "partition",
    "partition_copy",
    "stable_partition",
    "sort",
    "stable_sort",
    "is_sorted",
    "is_sorted_until",
    "is_heap",
    "is_heap_until",
    "max_element",
    "min_element",
    "minmax_element",
    "uninitialized_copy",
    "uninitialized_fill",
    "uninitialized_move",
    "uninitialized_default_construct",
    "uninitialized_value_construct",
    "destroy",
};

static constexpr const char *TwoRangeWithExecNames[] = {
    "partial_sort_copy",
    "includes",
    "set_union",
    "set_intersection",
    "set_difference",
    "set_symmetric_difference",
    "merge",
    "lexicographical_compare",
    "find_end",
    "search",
};

static constexpr const char *OneOrTwoRangeNames[] = {
    "is_permutation",
};

static constexpr const char *OneOrTwoRangeWithExecNames[] = {
    "equal",
    "mismatch",
};

namespace {
class StdReplacer : public utils::UseRangesCheck::Replacer {
public:
  explicit StdReplacer(SmallVector<UseRangesCheck::Signature> Indexes)
      : Indexes(std::move(Indexes)) {}
  std::string getReplaceName(const NamedDecl &OriginalName) const override {
    return ("std::ranges::" + OriginalName.getName()).str();
  }
  ArrayRef<UseRangesCheck::Signature>
  getReplacementSignatures() const override {
    return Indexes;
  }
  std::optional<std::string>
  getHeaderInclusion(const NamedDecl & /*OriginalName*/) const override {
    return "<algorithm>";
  }

private:
  SmallVector<UseRangesCheck::Signature> Indexes;
};
} // namespace

utils::UseRangesCheck::ReplacerMap UseRangesCheck::getReplacerMap() const {

  utils::UseRangesCheck::ReplacerMap Result;

  // template<typename Iter> Func(Iter first, Iter last,...).
  static const Signature SingleRangeArgs = {{0}};
  // template<typename Policy, typename Iter>
  // Func(Policy policy, Iter first, // Iter last,...).
  static const Signature SingleRangeExecPolicy = {{1}};
  // template<typename Iter1, typename Iter2>
  // Func(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2,...).
  static const Signature TwoRangeArgs = {{0}, {2}};
  // template<typename Policy, typename Iter1, typename Iter2>
  // Func(Policy policy, Iter1 first1, Iter1 last1, Iter2 first2, Iter2
  // last2,...).
  static const Signature TwoRangeExecPolicy = {{1}, {3}};

  static const Signature SingleRangeFunc[] = {SingleRangeArgs};

  static const Signature SingleRangeExecFunc[] = {SingleRangeArgs,
                                                  SingleRangeExecPolicy};
  static const Signature TwoRangeExecFunc[] = {TwoRangeArgs,
                                               TwoRangeExecPolicy};
  static const Signature OneOrTwoFunc[] = {SingleRangeArgs, TwoRangeArgs};
  static const Signature OneOrTwoExecFunc[] = {
      SingleRangeArgs, SingleRangeExecPolicy, TwoRangeArgs, TwoRangeExecPolicy};

  static const std::pair<ArrayRef<Signature>, ArrayRef<const char *>> Names[] =
      {{SingleRangeFunc, SingleRangeNames},
       {SingleRangeExecFunc, SingleRangeWithExecNames},
       {TwoRangeExecFunc, TwoRangeWithExecNames},
       {OneOrTwoFunc, OneOrTwoRangeNames},
       {OneOrTwoExecFunc, OneOrTwoRangeWithExecNames}};
  SmallString<64> Buff;
  for (const auto &[Signature, Values] : Names) {
    auto Replacer = llvm::makeIntrusiveRefCnt<StdReplacer>(
        SmallVector<UseRangesCheck::Signature>{Signature.begin(),
                                               Signature.end()});
    for (const auto &Name : Values) {
      Buff.assign({"::std::", Name});
      Result.try_emplace(Buff, Replacer);
    }
  }
  return Result;
}

bool UseRangesCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus20;
}
} // namespace clang::tidy::modernize
