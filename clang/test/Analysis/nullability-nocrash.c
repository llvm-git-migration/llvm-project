// RUN: %clang_analyze_cc1 -w -analyzer-checker=nullability \
// RUN:                       -analyzer-output=text -verify %s
//
// expected-no-diagnostics
