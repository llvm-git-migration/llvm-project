// RUN: %clang_cc1 -fsyntax-only -verify -verify-ignore-unexpected=note %s

struct B {
  void operator==(B);
};

struct C {
  void operator==(C);
};

struct D {
  void operator==(D);
};

struct E : C, B {
  using C::operator==;
  using B::operator==;
};

struct F : D, E {};

void f() {
  F{} == F{};
  // expected-error@-1 {{member 'operator==' found in multiple base classes of different types}}
  // expected-error@-2 {{use of overloaded operator '==' is ambiguous}}
}
