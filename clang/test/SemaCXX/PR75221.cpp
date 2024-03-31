// RUN: %clang_cc1 -verify -std=c++11 -fsyntax-only %s
// expected-no-diagnostics

template <class T> using foo = struct foo {
  T size = 0;
};
foo a;
