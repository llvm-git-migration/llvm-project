// RUN: %clang_cc1 -fsyntax-only -verify %s -Wmissing-noreturn -Wreturn-type

namespace GH63009 {
struct S1 {
  [[noreturn]] S1();
};

int foo();

int test_1() {
  S1 s1;
  foo();
} // expected-error{{non-void function does not return a value}}
}
