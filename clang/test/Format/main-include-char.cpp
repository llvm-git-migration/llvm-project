// RUN: clang-format %s -style="{}" | FileCheck %s -check-prefix=QUOTE
// RUN: clang-format %s -style="{MainIncludeChar: Quote}" | FileCheck %s -check-prefix=QUOTE
// RUN: clang-format %s -style="{MainIncludeChar: Bracket}" | FileCheck %s -check-prefix=BRACKET

#include <a>
#include "quote/main-include-char.hpp"
#include <bracket/main-include-char.hpp>

// QUOTE: "quote/main-include-char.hpp"
// QUOTE-NEXT: <a>
// QUOTE-NEXT: <bracket/main-include-char.hpp>

// BRACKET: <bracket/main-include-char.hpp>
// BRACKET-NEXT: "quote/main-include-char.hpp"
// BRACKET-NEXT: <a>
