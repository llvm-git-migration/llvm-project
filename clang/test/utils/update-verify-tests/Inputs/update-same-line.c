// RUN: %clang_cc1 -verify %s
// RUN: diff %s %s.expected
void foo() {
    bar = 2;     //   expected-error       {{asdf}}
}

