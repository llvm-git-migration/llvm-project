// RUN: %clang_analyze_cc1 -w -analyzer-checker=nullability \
// RUN:                       -analyzer-output=text -verify %s
//
// expected-no-diagnostics
//
// This case previously crashed because of an assert in CheckerManager.cpp,
// checking for registered event dispatchers. This check is too strict so
// was removed by this commit. This test case covers the previous crash,
// and is expected to simply not crash. The source file can be anything,
// and does not need to be empty.
