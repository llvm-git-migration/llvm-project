// Trivial check to ensure skip-function-bodies flag is propagated.
//
// RUN: %clang_cc1 -verify -skip-function-bodies -pedantic-errors %s

int f() {
  // normally this should emit some diags, but we're skipping it!
  this is garbage; 
}
