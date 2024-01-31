// RUN: %clang_cc1 -triple amdgcn-- -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-feature +wavefrontsize32 -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1010 -target-feature +wavefrontsize32 -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1010 -target-feature -wavefrontsize64 -verify -S -o - %s
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1010 -verify -S -o - %s

// expected-no-diagnostics

typedef unsigned long ulong;

void test_ballot_wave64(global ulong* out, int a, int b) {
  *out = __builtin_amdgcn_ballot_w64(a == b);
}
