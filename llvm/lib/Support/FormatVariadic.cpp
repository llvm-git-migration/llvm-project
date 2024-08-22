//===- FormatVariadic.cpp - Format string parsing and analysis ----*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#include "llvm/Support/FormatVariadic.h"
#include "llvm/ADT/SmallSet.h"
#include <cassert>
#include <optional>
#include <variant>

using namespace llvm;

static std::optional<AlignStyle> translateLocChar(char C) {
  switch (C) {
  case '-':
    return AlignStyle::Left;
  case '=':
    return AlignStyle::Center;
  case '+':
    return AlignStyle::Right;
  default:
    return std::nullopt;
  }
  LLVM_BUILTIN_UNREACHABLE;
}

static bool consumeFieldLayout(StringRef &Spec, AlignStyle &Where,
                               size_t &Align, char &Pad) {
  Where = AlignStyle::Right;
  Align = 0;
  Pad = ' ';
  if (Spec.empty())
    return true;

  if (Spec.size() > 1) {
    // A maximum of 2 characters at the beginning can be used for something
    // other than the width.
    // If Spec[1] is a loc char, then Spec[0] is a pad char and Spec[2:...]
    // contains the width.
    // Otherwise, if Spec[0] is a loc char, then Spec[1:...] contains the width.
    // Otherwise, Spec[0:...] contains the width.
    if (auto Loc = translateLocChar(Spec[1])) {
      Pad = Spec[0];
      Where = *Loc;
      Spec = Spec.drop_front(2);
    } else if (auto Loc = translateLocChar(Spec[0])) {
      Where = *Loc;
      Spec = Spec.drop_front(1);
    }
  }

  bool Failed = Spec.consumeInteger(0, Align);
  return !Failed;
}

static std::variant<ReplacementItem, StringRef>
parseReplacementItem(StringRef Spec) {
  StringRef RepString = Spec.trim("{}");

  // If the replacement sequence does not start with a non-negative integer,
  // this is an error.
  char Pad = ' ';
  std::size_t Align = 0;
  AlignStyle Where = AlignStyle::Right;
  StringRef Options;
  size_t Index = 0;
  RepString = RepString.trim();
  if (RepString.consumeInteger(0, Index))
    return "Invalid replacement sequence index!";
  RepString = RepString.trim();
  if (RepString.consume_front(",")) {
    if (!consumeFieldLayout(RepString, Where, Align, Pad))
      return "Invalid replacement field layout specification!";
  }
  RepString = RepString.trim();
  if (RepString.consume_front(":")) {
    Options = RepString.trim();
    RepString = StringRef();
  }
  RepString = RepString.trim();
  if (!RepString.empty())
    return "Unexpected character found in replacement string!";
  return ReplacementItem{Spec, Index, Align, Where, Pad, Options};
}

static std::variant<std::pair<ReplacementItem, StringRef>, StringRef>
splitLiteralAndReplacement(StringRef Fmt) {
  // Everything up until the first brace is a literal.
  if (Fmt.front() != '{') {
    std::size_t BO = Fmt.find_first_of('{');
    return std::make_pair(ReplacementItem(Fmt.substr(0, BO)), Fmt.substr(BO));
  }

  StringRef Braces = Fmt.take_while([](char C) { return C == '{'; });
  // If there is more than one brace, then some of them are escaped.  Treat
  // these as replacements.
  if (Braces.size() > 1) {
    size_t NumEscapedBraces = Braces.size() / 2;
    StringRef Middle = Fmt.take_front(NumEscapedBraces);
    StringRef Right = Fmt.drop_front(NumEscapedBraces * 2);
    return std::make_pair(ReplacementItem(Middle), Right);
  }
  // An unterminated open brace is undefined. Assert to indicate that this is
  // undefined and that we consider it an error. When asserts are disabled,
  // build a replacement item with an error message.
  std::size_t BC = Fmt.find_first_of('}');
  if (BC == StringRef::npos)
    return "Unterminated brace sequence. Escape with {{ for a literal brace.";

  // Even if there is a closing brace, if there is another open brace before
  // this closing brace, treat this portion as literal, and try again with the
  // next one.
  std::size_t BO2 = Fmt.find_first_of('{', 1);
  if (BO2 < BC)
    return std::make_pair(ReplacementItem{Fmt.substr(0, BO2)}, Fmt.substr(BO2));

  StringRef Spec = Fmt.slice(1, BC);
  StringRef Right = Fmt.substr(BC + 1);

  auto RI = parseReplacementItem(Spec);
  if (const StringRef *ErrMsg = std::get_if<1>(&RI))
    return *ErrMsg;

  return std::make_pair(std::get<0>(RI), Right);
}

std::pair<SmallVector<ReplacementItem, 2>, bool>
formatv_object_base::parseFormatString(raw_ostream &S, const StringRef Fmt,
                                       size_t NumArgs, bool Validate) {
  SmallVector<ReplacementItem, 2> Replacements;
  ReplacementItem I;
  size_t NumExpectedArgs = 0;

  // Make a copy for pasring as it updates it.
  StringRef ParseFmt = Fmt;
  while (!ParseFmt.empty()) {
    auto RI = splitLiteralAndReplacement(ParseFmt);
    if (const StringRef *ErrMsg = std::get_if<1>(&RI)) {
      // If there was an error parsing the format string, write the error to the
      // stream, and return false as second member of the pair.
      errs() << "Invalid format string: " << Fmt << "\n";
      assert(0 && "Invalid format string");
      S << *ErrMsg;
      return {{}, false};
    }
    std::tie(I, ParseFmt) = std::get<0>(RI);
    if (I.Type != ReplacementType::Empty)
      Replacements.push_back(I);
    if (I.Type == ReplacementType::Format)
      NumExpectedArgs = std::max(NumExpectedArgs, I.Index + 1);
  }
  if (!Validate)
    return {Replacements, true};

  // Perform additional validation. Verify that the number of arguments matches
  // the number of replacement indices and that there are no holes in the
  // replacement indexes.
  if (NumExpectedArgs != NumArgs) {
    errs() << "Expected " << NumExpectedArgs << " Args, but got " << NumArgs
           << " for format string '" << Fmt << "'\n";
    assert(0 && "Invalid formatv() call");
    S << "Expected " << NumExpectedArgs << " Args, but got " << NumArgs
      << " for format string '" << Fmt << "'\n";
    return {{}, false};
  }

  SmallSet<size_t, 2> Indices;
  for (const ReplacementItem &R : Replacements) {
    if (R.Type != ReplacementType::Format)
      continue;
    Indices.insert(R.Index);
  }

  if (Indices.size() != NumExpectedArgs) {
    errs() << "Invalid format string: Replacement field indices "
              "cannot have holes for format string '"
           << Fmt << "'\n";
    assert(0 && "Invalid format string");
    S << "Replacement field indices cannot have holes for format string '"
      << Fmt << "'\n";
    return {{}, false};
  }

  return {Replacements, true};
}

void support::detail::format_adapter::anchor() {}
