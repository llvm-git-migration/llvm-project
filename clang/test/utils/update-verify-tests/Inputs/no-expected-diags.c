// RUN: %clang_cc1 -verify %s
// RUN: diff %s %s.expected
// expected-no-diagnostics
void foo() {
    a = 2;
}
