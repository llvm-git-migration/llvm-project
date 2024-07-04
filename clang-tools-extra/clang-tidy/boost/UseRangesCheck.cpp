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
#include <initializer_list>
#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::boost {

namespace {
/// Base replacer that handles the boost include path and namespace
class BoostReplacer : public UseRangesCheck::Replacer {
public:
  BoostReplacer(ArrayRef<UseRangesCheck::Signature> Signatures,
                bool IncludeSystem)
      : Signature(Signatures), IncludeSystem(IncludeSystem) {}

  ArrayRef<UseRangesCheck::Signature> getReplacementSignatures() const final {
    return Signature;
  }

  virtual std::pair<StringRef, StringRef>
  getBoostName(const NamedDecl &OriginalName) const = 0;
  virtual std::pair<StringRef, StringRef>
  getBoostHeader(const NamedDecl &OriginalName) const = 0;

  std::string getReplaceName(const NamedDecl &OriginalName) const final {
    auto [Namespace, Function] = getBoostName(OriginalName);
    return ("boost::" + Namespace + (Namespace.empty() ? "" : "::") + Function)
        .str();
  }

  std::optional<std::string>
  getHeaderInclusion(const NamedDecl &OriginalName) const final {
    auto [Path, HeaderName] = getBoostHeader(OriginalName);
    return ((IncludeSystem ? "<boost/" : "boost/") + Path +
            (Path.empty() ? "" : "/") + HeaderName +
            (IncludeSystem ? ".hpp>" : ".hpp"))
        .str();
  }

private:
  SmallVector<UseRangesCheck::Signature> Signature;
  bool IncludeSystem;
};

/// Creates replaces where the header file lives in
/// `boost/algorithm/<FUNC_NAME>.hpp and the function is named
/// `boost::range::<FUNC_NAME>`
class BoostRangeAlgorithmReplacer : public BoostReplacer {
public:
  using BoostReplacer::BoostReplacer;
  std::pair<StringRef, StringRef>
  getBoostName(const NamedDecl &OriginalName) const override {
    return {"range", OriginalName.getName()};
  }

  std::pair<StringRef, StringRef>
  getBoostHeader(const NamedDecl &OriginalName) const override {
    return {"range/algorithm", OriginalName.getName()};
  }
};

/// Creates replaces where the header file lives in
/// `boost/algorithm/<CUSTOM_HEADER>.hpp and the function is named
/// `boost::range::<FUNC_NAME>`
class CustomBoostAlgorithmHeaderReplacer : public BoostRangeAlgorithmReplacer {
public:
  CustomBoostAlgorithmHeaderReplacer(
      StringRef HeaderName, ArrayRef<UseRangesCheck::Signature> Signature,
      bool IncludeSystem)
      : BoostRangeAlgorithmReplacer(Signature, IncludeSystem),
        HeaderName(HeaderName) {}

  std::pair<StringRef, StringRef>
  getBoostHeader(const NamedDecl & /*OriginalName*/) const override {
    return {"range/algorithm", HeaderName};
  }

private:
  StringRef HeaderName;
};

/// Creates replaces where the header file lives in
/// `boost/algorithm/<SUB_HEADER>.hpp and the function is named
/// `boost::algorithm::<FUNC_NAME>`
class BoostAlgorithmReplacer : public BoostReplacer {
public:
  BoostAlgorithmReplacer(StringRef SubHeader,
                         ArrayRef<UseRangesCheck::Signature> Signature,
                         bool IncludeSystem)
      : BoostReplacer(Signature, IncludeSystem),
        SubHeader(("algorithm/" + SubHeader).str()) {}
  std::pair<StringRef, StringRef>
  getBoostName(const NamedDecl &OriginalName) const override {
    return {"algorithm", OriginalName.getName()};
  }

  std::pair<StringRef, StringRef>
  getBoostHeader(const NamedDecl &OriginalName) const override {
    return {SubHeader, OriginalName.getName()};
  }

  std::string SubHeader;
};

/// Creates replaces where the header file lives in
/// `boost/algorithm/<SUB_HEADER>/<HEADER_NAME>.hpp and the function is named
/// `boost::algorithm::<FUNC_NAME>`
class CustomBoostAlgorithmReplacer : public BoostReplacer {
public:
  CustomBoostAlgorithmReplacer(StringRef SubHeader, StringRef HeaderName,
                               ArrayRef<UseRangesCheck::Signature> Signature,
                               bool IncludeSystem)
      : BoostReplacer(Signature, IncludeSystem),
        SubHeader(("algorithm/" + SubHeader).str()), HeaderName(HeaderName) {}
  std::pair<StringRef, StringRef>
  getBoostName(const NamedDecl &OriginalName) const override {
    return {"algorithm", OriginalName.getName()};
  }

  std::pair<StringRef, StringRef>
  getBoostHeader(const NamedDecl & /*OriginalName*/) const override {
    return {SubHeader, HeaderName};
  }

  std::string SubHeader;
  StringRef HeaderName;
};

} // namespace

utils::UseRangesCheck::ReplacerMap UseRangesCheck::getReplacerMap() const {

  ReplacerMap Results;
  static const Signature SingleSig = {{0}};
  static const Signature TwoSig = {{0}, {2}};
  static const auto Add =
      [&Results](llvm::IntrusiveRefCntPtr<BoostReplacer> Replacer,
                 std::initializer_list<StringRef> Names) {
        for (const auto &Name : Names) {
          Results.try_emplace(("::std::" + Name).str(), Replacer);
        }
      };

  Add(llvm::makeIntrusiveRefCnt<CustomBoostAlgorithmHeaderReplacer>(
          "set_algorithm", TwoSig, IncludeBoostSystem),
      {"includes", "set_union", "set_intersection", "set_difference",
       "set_symmetric_difference"});
  Add(llvm::makeIntrusiveRefCnt<BoostRangeAlgorithmReplacer>(
          SingleSig, IncludeBoostSystem),
      {"unique",         "lower_bound",   "stable_sort",
       "equal_range",    "remove_if",     "sort",
       "random_shuffle", "remove_copy",   "stable_partition",
       "remove_copy_if", "count",         "copy_backward",
       "reverse_copy",   "adjacent_find", "remove",
       "upper_bound",    "binary_search", "replace_copy_if",
       "for_each",       "generate",      "count_if",
       "min_element",    "reverse",       "replace_copy",
       "fill",           "unique_copy",   "transform",
       "copy",           "replace",       "find",
       "replace_if",     "find_if",       "partition",
       "max_element"});
  Add(llvm::makeIntrusiveRefCnt<BoostRangeAlgorithmReplacer>(
          TwoSig, IncludeBoostSystem),
      {"find_end", "merge", "partial_sort_copy", "find_first_of", "search",
       "lexicographical_compare", "equal", "mismatch"});
  Add(llvm::makeIntrusiveRefCnt<CustomBoostAlgorithmHeaderReplacer>(
          "permutation", SingleSig, IncludeBoostSystem),
      {"next_permutation", "prev_permutation"});
  Add(llvm::makeIntrusiveRefCnt<CustomBoostAlgorithmHeaderReplacer>(
          "heap_algorithm", SingleSig, IncludeBoostSystem),
      {"push_heap", "pop_heap", "make_heap", "sort_heap"});
  Add(llvm::makeIntrusiveRefCnt<BoostAlgorithmReplacer>("cxx11", SingleSig,
                                                        IncludeBoostSystem),
      {"copy_if", "is_permutation", "is_partitioned", "find_if_not",
       "partition_copy", "any_of", "iota", "all_of", "partition_point",
       "is_sorted", "none_of"});
  Add(llvm::makeIntrusiveRefCnt<CustomBoostAlgorithmReplacer>(
          "cxx11", "is_sorted", SingleSig, IncludeBoostSystem),
      {"is_sorted_until"});
  Add(llvm::makeIntrusiveRefCnt<BoostAlgorithmReplacer>("cxx17", SingleSig,
                                                        IncludeBoostSystem),
      {"reduce"});

  return Results;
}

UseRangesCheck::UseRangesCheck(StringRef Name, ClangTidyContext *Context)
    : utils::UseRangesCheck(Name, Context),
      IncludeBoostSystem(Options.get("IncludeBoostSystem", true)) {}

void UseRangesCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  utils::UseRangesCheck::storeOptions(Opts);
  Options.store(Opts, "IncludeBoostSystem", IncludeBoostSystem);
}
DiagnosticBuilder UseRangesCheck::createDiag(const CallExpr &Call) {
  return diag(Call.getBeginLoc(), "use a boost version of this algorithm");
}
} // namespace clang::tidy::boost
