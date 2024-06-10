// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -fsyntax-only -verify -std=gnu++11 -triple amdgcn -Wno-unused-value %s

void foo() {
  int n = 100;
  __buffer_rsrc_t v = 0; // expected-error {{cannot initialize a variable of type '__buffer_rsrc_t' with an rvalue of type 'int'}}
  static_cast<__buffer_rsrc_t>(n); // expected-error {{static_cast from 'int' to '__buffer_rsrc_t' is not allowed}}
  dynamic_cast<__buffer_rsrc_t>(n); // expected-error {{invalid target type '__buffer_rsrc_t' for dynamic_cast; target type must be a reference or pointer type to a defined class}}
  reinterpret_cast<__buffer_rsrc_t>(n); // expected-error {{reinterpret_cast from 'int' to '__buffer_rsrc_t' is not allowed}}
  int c(v); // expected-error {{cannot initialize a variable of type 'int' with an lvalue of type '__buffer_rsrc_t'}}
  sizeof(__buffer_rsrc_t); // expected-error {{invalid application of 'sizeof' to sizeless type '__buffer_rsrc_t'}}
  alignof(__buffer_rsrc_t); // expected-error {{invalid application of 'alignof' to sizeless type '__buffer_rsrc_t'}}
  static __buffer_rsrc_t table[1]; // expected-error {{array has sizeless element type '__buffer_rsrc_t'}}
  __buffer_rsrc_t k;
  int *ip = (int *)k; // expected-error {{cannot cast from type '__buffer_rsrc_t' to pointer type 'int *'}}
}
