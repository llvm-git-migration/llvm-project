// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -fsyntax-only -verify -triple amdgcn -Wno-unused-value %s

void foo() {
  int n = 100;
  __buffer_rsrc_t v = 0; // expected-error {{cannot initialize a variable of type '__buffer_rsrc_t' with an rvalue of type 'int'}}
  static_cast<__buffer_rsrc_t>(n); // expected-error {{static_cast from 'int' to '__buffer_rsrc_t' is not allowed}}
  dynamic_cast<__buffer_rsrc_t>(n); // expected-error {{invalid target type '__buffer_rsrc_t' for dynamic_cast; target type must be a reference or pointer type to a defined class}}
  reinterpret_cast<__buffer_rsrc_t>(n); // expected-error {{reinterpret_cast from 'int' to '__buffer_rsrc_t' is not allowed}}
  int c(v); // expected-error {{cannot initialize a variable of type 'int' with an lvalue of type '__buffer_rsrc_t'}}
}
