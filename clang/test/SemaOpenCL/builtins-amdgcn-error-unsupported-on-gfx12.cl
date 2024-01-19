// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu gfx1200 -verify -S -o - %s

#pragma OPENCL EXTENSION cl_khr_fp64:enable

typedef float  v2f   __attribute__((ext_vector_type(2)));
typedef float  v4f   __attribute__((ext_vector_type(4)));
typedef float  v16f  __attribute__((ext_vector_type(16)));
typedef float  v32f  __attribute__((ext_vector_type(32)));
typedef half   v4h   __attribute__((ext_vector_type(4)));
typedef half   v8h   __attribute__((ext_vector_type(8)));
typedef half   v16h  __attribute__((ext_vector_type(16)));
typedef half   v32h  __attribute__((ext_vector_type(32)));
typedef int    v2i   __attribute__((ext_vector_type(2)));
typedef int    v4i   __attribute__((ext_vector_type(4)));
typedef int    v16i  __attribute__((ext_vector_type(16)));
typedef int    v32i  __attribute__((ext_vector_type(32)));
typedef short  v2s   __attribute__((ext_vector_type(2)));
typedef short  v4s   __attribute__((ext_vector_type(4)));
typedef short  v8s   __attribute__((ext_vector_type(8)));
typedef short  v16s  __attribute__((ext_vector_type(16)));
typedef short  v32s  __attribute__((ext_vector_type(32)));
typedef double v4d   __attribute__((ext_vector_type(4)));

void test(global v32f*    out_v32f,
          global v16f*    out_v16f,
	  global v4f*     out_v4f,
	  global v32i*    out_v32i,
	  global v16i*    out_v16i,
	  global v4i*     out_v4i,
	  global v4d*     out_v4d,
	  global double*  out_double,
	  double a_double , double b_double , double c_double,
          float a_float   , float  b_float  , float  c_float,
	  int   a_int     , int    b_int    , int    c_int,
	  long  a_long    , long   b_long   , long   c_long,
	  v4d   a_v4d     , v4d    b_v4d    , v4d    c_v4d,
	  v8s   a_v8s     , v8s    b_v8s    , v8s    c_v8s,
	  v4s   a_v4s     , v4s    b_v4s    , v4s    c_v4s,
	  v2s   a_v2s     , v2s    b_v2s    , v2s    c_v2s,
	  v2i   a_v2i     , v2i    b_v2i    , v2i    c_v2i,
	  v16i  a_v16i    , v16i   b_v16i   , v16i   c_v16i,
	  v32i  a_v32i    , v32i   b_v32i   , v32i   c_v32i,
	  v4i   a_v4i     , v4i    b_v4i    , v4i    c_v4i,
	  v2f   a_v2f     , v2f    b_v2f    , v2f    c_v2f,
	  v4f   a_v4f     , v4f    b_v4f    , v4f    c_v4f,
	  v16f  a_v16f    , v16f   b_v16f   , v16f   c_v16f,
	  v32f  a_v32f    , v32f   b_v32f   , v32f   c_v32f,
	  v4h   a_v4h     , v4h    b_v4h    , v4h    c_v4h,
	  v8h   a_v8h     , v8h    b_v8h    , v8h    c_v8h,
	  int   idx) {
  *out_v32f = __builtin_amdgcn_mfma_f32_32x32x1f32(a_float, b_float, c_v32f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x1f32' needs target feature mai-insts}}
  *out_v16f = __builtin_amdgcn_mfma_f32_16x16x1f32(a_float, b_float, c_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x1f32' needs target feature mai-insts}}
  *out_v4f =  __builtin_amdgcn_mfma_f32_4x4x1f32(a_float, b_float, c_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_4x4x1f32' needs target feature mai-insts}}
  *out_v16f = __builtin_amdgcn_mfma_f32_32x32x2f32(a_float, b_float, c_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x2f32' needs target feature mai-insts}}
  *out_v4f =  __builtin_amdgcn_mfma_f32_16x16x4f32(a_float, b_float, c_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x4f32' needs target feature mai-insts}}
  *out_v32f = __builtin_amdgcn_mfma_f32_32x32x4f16(a_v4h, b_v4h, c_v32f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x4f16' needs target feature mai-insts}}
  *out_v16f = __builtin_amdgcn_mfma_f32_16x16x4f16(a_v4h, b_v4h, c_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x4f16' needs target feature mai-insts}}
  *out_v4f = __builtin_amdgcn_mfma_f32_4x4x4f16(a_v4h, b_v4h, c_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_4x4x4f16' needs target feature mai-insts}}
  *out_v16f = __builtin_amdgcn_mfma_f32_32x32x8f16(a_v4h, b_v4h, c_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x8f16' needs target feature mai-insts}}
  *out_v4f = __builtin_amdgcn_mfma_f32_16x16x16f16(a_v4h, b_v4h, c_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x16f16' needs target feature mai-insts}}
  *out_v32i = __builtin_amdgcn_mfma_i32_32x32x4i8(a_int, b_int, c_v32i, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_i32_32x32x4i8' needs target feature mai-insts}}
  *out_v16i = __builtin_amdgcn_mfma_i32_16x16x4i8(a_int, b_int, c_v16i, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_i32_16x16x4i8' needs target feature mai-insts}}
  *out_v4i = __builtin_amdgcn_mfma_i32_4x4x4i8(a_int, b_int, c_v4i, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_i32_4x4x4i8' needs target feature mai-insts}}
  *out_v16i = __builtin_amdgcn_mfma_i32_32x32x8i8(a_int, b_int, c_v16i, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_i32_32x32x8i8' needs target feature mai-insts}}
  *out_v4i = __builtin_amdgcn_mfma_i32_16x16x16i8(a_int, b_int, c_v4i, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_i32_16x16x16i8' needs target feature mai-insts}}
  *out_v32f = __builtin_amdgcn_mfma_f32_32x32x2bf16(a_v2s, b_v2s, c_v32f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x2bf16' needs target feature mai-insts}}
  *out_v16f = __builtin_amdgcn_mfma_f32_16x16x2bf16(a_v2s, b_v2s, c_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x2bf16' needs target feature mai-insts}}
  *out_v4f = __builtin_amdgcn_mfma_f32_4x4x2bf16(a_v2s, b_v2s, c_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_4x4x2bf16' needs target feature mai-insts}}
  *out_v16f = __builtin_amdgcn_mfma_f32_32x32x4bf16(a_v2s, b_v2s, c_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x4bf16' needs target feature mai-insts}}
  *out_v4f = __builtin_amdgcn_mfma_f32_16x16x8bf16(a_v2s, b_v2s, c_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x8bf16' needs target feature mai-insts}}
  *out_v32f = __builtin_amdgcn_mfma_f32_32x32x4bf16_1k(a_v4s, b_v4s, c_v32f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x4bf16_1k' needs target feature mai-insts}}
  *out_v16f = __builtin_amdgcn_mfma_f32_16x16x4bf16_1k(a_v4s, b_v4s, c_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x4bf16_1k' needs target feature mai-insts}}
  *out_v4f = __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(a_v4s, b_v4s, c_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_4x4x4bf16_1k' needs target feature mai-insts}}
  *out_v16f = __builtin_amdgcn_mfma_f32_32x32x8bf16_1k(a_v4s, b_v4s, c_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x8bf16_1k' needs target feature mai-insts}}
  *out_v4f = __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a_v4s, b_v4s, c_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x16bf16_1k' needs target feature mai-insts}}
  *out_v4d = __builtin_amdgcn_mfma_f64_16x16x4f64(a_double, b_double, c_v4d, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f64_16x16x4f64' needs target feature mai-insts}}
  *out_double = __builtin_amdgcn_mfma_f64_4x4x4f64(a_double, b_double, c_double, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f64_4x4x4f64' needs target feature mai-insts}}
  *out_v4i = __builtin_amdgcn_mfma_i32_16x16x32_i8(a_long, b_long, c_v4i, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_i32_16x16x32_i8' needs target feature mai-insts}}
  *out_v16i = __builtin_amdgcn_mfma_i32_32x32x16_i8(a_long, b_long, c_v16i, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_i32_32x32x16_i8' needs target feature mai-insts}}
  *out_v4f = __builtin_amdgcn_mfma_f32_16x16x8_xf32(a_v2f, b_v2f, c_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x8_xf32' needs target feature mai-insts}}
  *out_v16f = __builtin_amdgcn_mfma_f32_32x32x4_xf32(a_v2f, b_v2f, c_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x4_xf32' needs target feature mai-insts}}
  *out_v4f = __builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8(a_long, b_long, c_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x32_bf8_bf8' needs target feature fp8-insts}}
  *out_v4f = __builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8(a_long, b_long, c_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x32_bf8_fp8' needs target feature fp8-insts}}
  *out_v4f = __builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8(a_long, b_long, c_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x32_fp8_bf8' needs target feature fp8-insts}}
  *out_v4f = __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a_long, b_long, c_v4f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8' needs target feature fp8-insts}}
  *out_v16f = __builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8(a_long, b_long, c_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x16_bf8_bf8' needs target feature fp8-insts}}
  *out_v16f = __builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8(a_long, b_long, c_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x16_bf8_fp8' needs target feature fp8-insts}}
  *out_v16f = __builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8(a_long, b_long, c_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x16_fp8_bf8' needs target feature fp8-insts}}
  *out_v16f = __builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8(a_long, b_long, c_v16f, 0, 0, 0); // expected-error {{'__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8' needs target feature fp8-insts}}
  *out_v4f = __builtin_amdgcn_smfmac_f32_16x16x32_f16(a_v4h, b_v8h, c_v4f, idx, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_16x16x32_f16' needs target feature mai-insts}}
  *out_v16f = __builtin_amdgcn_smfmac_f32_32x32x16_f16(a_v4h, b_v8h, c_v16f, idx, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_32x32x16_f16' needs target feature mai-insts}}
  *out_v4f = __builtin_amdgcn_smfmac_f32_16x16x32_bf16(a_v4s, b_v8s, c_v4f, idx, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_16x16x32_bf16' needs target feature mai-insts}}
  *out_v16f = __builtin_amdgcn_smfmac_f32_32x32x16_bf16(a_v4s, b_v8s, c_v16f, idx, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_32x32x16_bf16' needs target feature mai-insts}}
  *out_v4i = __builtin_amdgcn_smfmac_i32_16x16x64_i8(a_v2i, b_v4i, c_v4i, idx, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_i32_16x16x64_i8' needs target feature mai-insts}}
  *out_v16i = __builtin_amdgcn_smfmac_i32_32x32x32_i8(a_v2i, b_v4i, c_v16i, idx, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_i32_32x32x32_i8' needs target feature mai-insts}}
  *out_v4f = __builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8(a_v2i, b_v4i, c_v4f, idx, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_16x16x64_bf8_bf8' needs target feature fp8-insts}}
  *out_v4f = __builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8(a_v2i, b_v4i, c_v4f, idx, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_16x16x64_bf8_fp8' needs target feature fp8-insts}}
  *out_v4f = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8(a_v2i, b_v4i, c_v4f, idx, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_16x16x64_fp8_bf8' needs target feature fp8-insts}}
  *out_v4f = __builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8(a_v2i, b_v4i, c_v4f, idx, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_16x16x64_fp8_fp8' needs target feature fp8-insts}}
  *out_v16f = __builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8(a_v2i, b_v4i, c_v16f, idx, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_32x32x32_bf8_bf8' needs target feature fp8-insts}}
  *out_v16f = __builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8(a_v2i, b_v4i, c_v16f, idx, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_32x32x32_bf8_fp8' needs target feature fp8-insts}}
  *out_v16f = __builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8(a_v2i, b_v4i, c_v16f, idx, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_32x32x32_fp8_bf8' needs target feature fp8-insts}}
  *out_v16f = __builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8(a_v2i, b_v4i, c_v16f, idx, 0, 0); // expected-error {{'__builtin_amdgcn_smfmac_f32_32x32x32_fp8_fp8' needs target feature fp8-insts}}
}
