// RUN: %clang_cc1 -fsyntax-only -verify -pedantic-errors %s

extern "C" {
  void* main; // expected-error {{'main' cannot have linkage specification 'extern "C"'}}
}
