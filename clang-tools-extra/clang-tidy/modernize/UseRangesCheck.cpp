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
#include "llvm/ADT/StringRef.h"

namespace clang::tidy::modernize {

utils::UseRangesCheck::ReplacerMap UseRangesCheck::GetReplacerMap() const {
  class StdReplacer : public utils::UseRangesCheck::Replacer {
  public:
    explicit StdReplacer(ArrayRef<ArrayRef<Indexes>> Indexes)
        : Indexes(Indexes) {}
    std::string getReplaceName(const NamedDecl &OriginalName) const override {
      return ("std::ranges::" + OriginalName.getName()).str();
    }
    ArrayRef<ArrayRef<Indexes>> getReplacementSignatures() const override {
      return Indexes;
    }
    std::optional<std::string>
    getHeaderInclusion(const NamedDecl &OriginalName) const override {
      return "<algorithm>";
    }

  private:
    ArrayRef<ArrayRef<Indexes>> Indexes;
  };
  using Indexes = UseRangesCheck::Replacer::Indexes;
  // using Signatures = Signature[];
  utils::UseRangesCheck::ReplacerMap Result;
  // template<typename Iter> Func(Iter first, Iter last,...).
  static const Indexes SingleRangeArgs[] = {{0}};
  // template<typename Policy, typename Iter>
  // Func(Policy policy, Iter first, // Iter last,...).
  static const Indexes SingleRangeExecPolicy[] = {{1}};
  // template<typename Iter1, typename Iter2>
  // Func(Iter1 first1, Iter1 last1, Iter2 first2, Iter2 last2,...).
  static const Indexes TwoRangeArgs[] = {{0}, {2}};
  // template<typename Policy, typename Iter1, typename Iter2>
  // Func(Policy policy, Iter1 first1, Iter1 last1, Iter2 first2, Iter2
  // last2,...).
  static const Indexes TwoRangeExecPolicy[] = {{1}, {3}};

  static const ArrayRef<Indexes> SingleRangeFunc[] = {SingleRangeArgs};

  static const ArrayRef<Indexes> SingleRangeExecFunc[] = {
      SingleRangeArgs, SingleRangeExecPolicy};
  static const ArrayRef<Indexes> TwoRangeExecFunc[] = {TwoRangeArgs,
                                                       TwoRangeExecPolicy};
  static const ArrayRef<Indexes> OneOrTwoFunc[] = {SingleRangeArgs,
                                                   TwoRangeArgs};
  static const ArrayRef<Indexes> OneOrTwoExecFunc[] = {
      SingleRangeArgs, SingleRangeExecPolicy, TwoRangeArgs, TwoRangeExecPolicy};

  static const std::pair<ArrayRef<ArrayRef<Indexes>>, ArrayRef<StringRef>>
      Names[] = {
          {SingleRangeFunc,
           {"all_of",
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
            "iota"}},
          {SingleRangeExecFunc,
           {"reverse",
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
            "destroy"}},
          {TwoRangeExecFunc,
           {"partial_sort_copy", "includes", "set_union", "set_intersection",
            "set_difference", "set_symmetric_difference", "merge",
            "lexicographical_compare", "find_end", "search"}},
          {OneOrTwoFunc, {"is_permutation"}},
          {OneOrTwoExecFunc, {"equal", "mismatch"}}};
  SmallString<64> Buff;
  for (const auto &[Signature, Values] : Names) {
    auto Replacer = llvm::makeIntrusiveRefCnt<StdReplacer>(Signature);
    for (const auto &Name : Values) {
      Buff.clear();
      Result.try_emplace(("::std::" + Name).toStringRef(Buff), Replacer);
    }
  }
  return Result;
}

bool UseRangesCheck::isLanguageVersionSupported(
    const LangOptions &LangOpts) const {
  return LangOpts.CPlusPlus20;
}
} // namespace clang::tidy::modernize
