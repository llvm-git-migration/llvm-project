// Host compilation on ARM with neon enabled (no errors expected).
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature +neon -fsyntax-only -verify=quiet %s

// Host compilation on ARM with neon disabled.
// RUN: %clang_cc1 -triple aarch64-linux-gnu -target-feature -neon -fsyntax-only -verify %s

// Device compilation on ARM (no errors expected).
// RUN: %clang_cc1 -triple spirv64 -aux-triple aarch64-linux-gnu -fsycl-is-device -fsyntax-only -verify=quiet %s

// quiet-no-diagnostics
typedef __attribute__((neon_vector_type(4))) float float32x4_t;
// expected-error@-1 {{'neon_vector_type' attribute is not supported on targets missing 'neon', 'mve', 'sve' or 'sme'}}
typedef unsigned char poly8_t;
typedef __attribute__((neon_polyvector_type(8))) poly8_t poly8x8_t;
// expected-error@-1 {{'neon_polyvector_type' attribute is not supported on targets missing 'neon' or 'mve'}}
