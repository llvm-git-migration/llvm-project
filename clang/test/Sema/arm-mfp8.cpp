// RUN: %clang_cc1 -fsyntax-only -verify=scalar,neon -triple aarch64-arm-none-eabi \
// RUN: -target-feature -fp8 -target-feature +neon %s

// REQUIRES: aarch64-registered-target

#include <arm_neon.h>

void test_vector(mfloat8x8_t a, mfloat8x8_t b, uint8x8_t c) {
  a + b;  // neon-error {{invalid operands to binary expression ('mfloat8x8_t' (vector of 8 'mfloat8_t' values) and 'mfloat8x8_t')}}
  a - b;  // neon-error {{invalid operands to binary expression ('mfloat8x8_t' (vector of 8 'mfloat8_t' values) and 'mfloat8x8_t')}}
  a * b;  // neon-error {{invalid operands to binary expression ('mfloat8x8_t' (vector of 8 'mfloat8_t' values) and 'mfloat8x8_t')}}
  a / b;  // neon-error {{invalid operands to binary expression ('mfloat8x8_t' (vector of 8 'mfloat8_t' values) and 'mfloat8x8_t')}}

  a + c;  // neon-error {{invalid operands to binary expression ('mfloat8x8_t' (vector of 8 'mfloat8_t' values) and 'uint8x8_t' (vector of 8 'uint8_t' values))}}
  a - c;  // neon-error {{invalid operands to binary expression ('mfloat8x8_t' (vector of 8 'mfloat8_t' values) and 'uint8x8_t' (vector of 8 'uint8_t' values))}}
  a * c;  // neon-error {{invalid operands to binary expression ('mfloat8x8_t' (vector of 8 'mfloat8_t' values) and 'uint8x8_t' (vector of 8 'uint8_t' values))}}
  a / c;  // neon-error {{invalid operands to binary expression ('mfloat8x8_t' (vector of 8 'mfloat8_t' values) and 'uint8x8_t' (vector of 8 'uint8_t' values))}}
  c + b;  // neon-error {{invalid operands to binary expression ('uint8x8_t' (vector of 8 'uint8_t' values) and 'mfloat8x8_t' (vector of 8 'mfloat8_t' values))}}
  c - b;  // neon-error {{invalid operands to binary expression ('uint8x8_t' (vector of 8 'uint8_t' values) and 'mfloat8x8_t' (vector of 8 'mfloat8_t' values))}}
  c * b;  // neon-error {{invalid operands to binary expression ('uint8x8_t' (vector of 8 'uint8_t' values) and 'mfloat8x8_t' (vector of 8 'mfloat8_t' values))}}
  c / b;  // neon-error {{invalid operands to binary expression ('uint8x8_t' (vector of 8 'uint8_t' values) and 'mfloat8x8_t' (vector of 8 'mfloat8_t' values))}}
}
