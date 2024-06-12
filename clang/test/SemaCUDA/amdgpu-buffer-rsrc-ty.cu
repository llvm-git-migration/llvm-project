// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -aux-triple amdgcn-amd-amdhsa %s -fsyntax-only -verify

#define __device__ __attribute__((device))

__device__ __buffer_rsrc_t test_buffer_rsrc_t_device() {} // expected-warning {{non-void function does not return a value}}
__buffer_rsrc_t test_buffer_rsrc_t_host() {} // expected-error {{'__buffer_rsrc_t' can only be used in device-side function}}
