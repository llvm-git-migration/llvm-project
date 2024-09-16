// RUN: %clang_analyze_cc1 -analyzer-checker=core,debug.ExprInspection -verify %s \
// RUN:  -analyzer-constraints=z3 

// REQUIRES: Z3
//
// This LIT covers a crash associated with this test.
// The expectation is to not crash!
//

long a;
void b() {
  long c;
  if (~(b && a)) // expected-warning {{address of function 'b' will always evaluate to 'true'}}
  // expected-note@-1 {{prefix with the address-of operator to silence this warning}}
    c ^= 0; // expected-warning {{The left expression of the compound assignment is an uninitialized value. The computed value will also be garbage}}
}
