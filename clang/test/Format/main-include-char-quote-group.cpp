// Test the combination of regrouped include directives, via regexes and
// priorities, with a main header included with quotes. Quotes for the main
// header being the default behavior, the first test does not specify it.

// RUN: clang-format %s -style="{IncludeBlocks: Regroup, IncludeCategories: [{Regex: 'lib-a', Priority: 1}, {Regex: 'lib-b', Priority: 2}, {Regex: 'lib-c', Priority: 3}]}" | tee delme | FileCheck %s
// RUN: clang-format %s -style="{MainIncludeChar: Quote, IncludeBlocks: Regroup, IncludeCategories: [{Regex: 'lib-a', Priority: 1}, {Regex: 'lib-b', Priority: 2}, {Regex: 'lib-c', Priority: 3}]}" | FileCheck %s

#include <lib-c/header-1.hpp>
#include <lib-c/header-2.hpp>
#include <lib-c/header-3.hpp>
#include <lib-b/header-1.hpp>
#include "lib-b/main-include-char-quote-group.hpp"
#include <lib-b/header-3.hpp>
#include <lib-a/header-1.hpp>
#include <lib-a/main-include-char-quote-group.hpp>
#include <lib-a/header-3.hpp>

// CHECK: "lib-b/main-include-char-quote-group.hpp"
// CHECK-EMPTY:
// CHECK-NEXT: <lib-a/header-1.hpp>
// CHECK-NEXT: <lib-a/header-3.hpp>
// CHECK-NEXT: <lib-a/main-include-char-quote-group.hpp>
// CHECK-EMPTY:
// CHECK-NEXT: <lib-b/header-1.hpp>
// CHECK-NEXT: <lib-b/header-3.hpp>
// CHECK-EMPTY:
// CHECK-NEXT: <lib-c/header-1.hpp>
// CHECK-NEXT: <lib-c/header-2.hpp>
// CHECK-NEXT: <lib-c/header-3.hpp>
