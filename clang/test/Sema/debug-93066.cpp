// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

struct S {
  void f() {
    ++this;
    // expected-error@-1 {{expression is not assignable}}
    // expected-note@-2 {{dereference the pointer to modify}}
  }

  void g() const {
    ++this;
    // expected-error@-1 {{expression is not assignable}}
  }
};
