// RUN: %clang_cc1 -verify %s
// RUN: diff %s %s.expected
void foo() {
    // expected-error@+1{{asdf}}
    bar = 2;
}
