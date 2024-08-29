// RUN: %clang_cc1 -verify %s
// RUN: diff %s %s.expected
void foo() {
    a = 2;
    b = 2;

    c = 2;
}
