// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -fsyntax-only -verify -std=gnu++11 -triple amdgcn -Wno-unused-value %s

void foo() {
  int n = 100;
  __amdgcn_buffer_rsrc_t v = 0; // expected-error {{cannot initialize a variable of type '__amdgcn_buffer_rsrc_t' with an rvalue of type 'int'}}
  static_cast<__amdgcn_buffer_rsrc_t>(n); // expected-error {{static_cast from 'int' to '__amdgcn_buffer_rsrc_t' is not allowed}}
  dynamic_cast<__amdgcn_buffer_rsrc_t>(n); // expected-error {{invalid target type '__amdgcn_buffer_rsrc_t' for dynamic_cast; target type must be a reference or pointer type to a defined class}}
  reinterpret_cast<__amdgcn_buffer_rsrc_t>(n); // expected-error {{reinterpret_cast from 'int' to '__amdgcn_buffer_rsrc_t' is not allowed}}
  int c(v); // expected-error {{cannot initialize a variable of type 'int' with an lvalue of type '__amdgcn_buffer_rsrc_t'}}
  __amdgcn_buffer_rsrc_t k;
  int *ip = (int *)k; // expected-error {{cannot cast from type '__amdgcn_buffer_rsrc_t' to pointer type 'int *'}}
  void *vp = (void *)k; // expected-error {{cannot cast from type '__amdgcn_buffer_rsrc_t' to pointer type 'void *'}}
}

static_assert(sizeof(__amdgcn_buffer_rsrc_t) == 16, "wrong size");
static_assert(alignof(__amdgcn_buffer_rsrc_t) == 16, "wrong aignment");
