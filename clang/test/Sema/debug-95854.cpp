// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s
//
// expected-no-diagnostics

struct A {
  union {
    int n = 0;
    int m;
  };
};
const A a;
